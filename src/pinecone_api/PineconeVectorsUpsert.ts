import { Vector as PineconeVector } from "@pinecone-database/pinecone";
import { PineconeUpsertOptions, PineconeUpsertResults, PineconeUpsertVectorsRecord, NodePineconeVectorMap } from "./types";
import { UpsertResponse as PineconeUpsertResponse, VectorOperationsApi as Index, UpsertResponse } from "@pinecone-database/pinecone/dist/pinecone-generated-ts-fetch";

const DEFAULT_UPSERT_BATCH_SIZE = 100;

/**
 * Class to handle upserting vectors into Pinecone. Extracted into its own
 * class for testability and because there is a lot going on.
 * 
 * @param {Index}: a Pinecone index where the vectors should be upserted
 * @param {options}: options for executing upserts with this instance. Supports
 *   batch size for batched requests. Future improvements include parallel execution.
 */
export class PineconeVectorsUpsert {
  pineconeIndex: Index;
  batchSize: number;


  DEFAULT_OPTIONS: PineconeUpsertOptions = {
    batchSize: DEFAULT_UPSERT_BATCH_SIZE,
    includeSparseValues: true
  }

  constructor(index: Index, options: PineconeUpsertOptions) {
    this.batchSize = options.batchSize || this.DEFAULT_OPTIONS.batchSize!;
    this.pineconeIndex = index;
  }

  async execute(pineconeVectorsByNode: PineconeUpsertVectorsRecord) {
    const batchSize = this.batchSize;
    if (pineconeVectorsByNode.totalVectorCount > (batchSize)) {
      return this.batchUpsert(pineconeVectorsByNode.vectorsByNode, batchSize);
    } else {
      return this.singleUpsert(pineconeVectorsByNode.vectorsByNode);
    }
  }

  // Upserts a single array of pinecone vectors. This assumes that the vectors are
  // less than or equal to the desired batch size.
  async singleUpsert(vectorsByNode: NodePineconeVectorMap): Promise<PineconeUpsertResults> {
    let upsertedCount = 0;

    // Since we are upserting all vectors at once, we need to flatten the vectors.
    const builtVectors = Object.entries(vectorsByNode).reduce((allVectors, [nodeId, vectors]) => {
      return allVectors.concat(vectors);
    }, [] as PineconeVector[]);

    try {
      const upsertResponse = await this.pineconeIndex.upsert({ upsertRequest: { vectors: builtVectors } });
      upsertedCount = upsertResponse.upsertedCount || 0;
    } catch (e) {
      throw `Error with call to Pinecone: ${e}`;
    }

    const upsertedNodeIds = Object.keys(vectorsByNode);
    const upsertResults = {
      upsertedNodeCount: upsertedNodeIds.length,
      upsertedNodeIds,
      upsertedVectorCount: builtVectors.length,
      upsertedVectorByNode: vectorsByNode,
      failedNodeCount: 0,
      failedNodeIds: [],
      errors: []
    };
    return Promise.resolve(upsertResults);
  }


  // Iterate through the vectorsByNode, collecting batches of vectors to upsert.
  // We group together as many vectors per node as possible, up to the batch size. Try to fit
  // entire nodes into a batch, then fill the rest of the batch with vectors from other nodes.
  buildVectorBatches(vectorsByNode: NodePineconeVectorMap, batchSize: number): Array<NodePineconeVectorMap> {
    let currentVectorBatch = {} as NodePineconeVectorMap;
    let currentVectorBatchSize = 0;
    // let nodeIdsProcessed = 0;

    const nodeVectorEntries = Object.entries(vectorsByNode);
    const builtVectorBatches = nodeVectorEntries.reduce((builtVectorBatches, [nodeId, vectors], currentIndex) => {

      // If adding the vectors from this node to the current batch would
      // not put us over the batch size, add them to the current batch.
      // Then move on to the next node.
      if (currentVectorBatchSize + vectors.length <= batchSize) {
        builtVectorBatches.push(currentVectorBatch);
        currentVectorBatchSize += vectors.length;
        return builtVectorBatches;
      }

      // Fill the current batch as much as possible with vectors from this node.
      const currentBatchSpaceRemaining = currentVectorBatchSize - batchSize
      currentVectorBatch[nodeId] = vectors.slice(0, currentBatchSpaceRemaining);

      // Now we start a new batch and reset the batch size counter.
      currentVectorBatch = { [nodeId]: [] } as NodePineconeVectorMap;
      currentVectorBatchSize = 0;

      // Now we loop over the remaining vectors, filling up batches and adding
      // them to the list
      for (const subVector of vectors.slice(currentBatchSpaceRemaining)) {
        // If adding this vector to the current batch would put us over
        // the batch size, add the current batch to the list of batches
        // and start a new batch.
        if (currentVectorBatchSize + 1 > batchSize) {
          builtVectorBatches.push(currentVectorBatch);
          currentVectorBatch = { [nodeId]: [] } as NodePineconeVectorMap;
          currentVectorBatchSize = 0;
        }
        currentVectorBatch[nodeId].concat(subVector);
        currentVectorBatchSize += 1;
      }

      return builtVectorBatches;
    }, [] as Array<NodePineconeVectorMap>);
    return builtVectorBatches;
  }

  /*
  * Batch upsert vectors into the index.
  *   @param {PineconeVector[]} builtVectors - an array of vectors to upsert.
  *   @param {number} batchSize - the size of each batch to upsert.
  *   @returns {Promise<number>} the number of vectors affected.
  */
  async batchUpsert(builtVectors: NodePineconeVectorMap, batchSize: number): Promise<PineconeUpsertResults> {
    const vectorBatches = this.buildVectorBatches(builtVectors, batchSize);

    // Accumulate an array of promises for all of the upserts we're doing.
    const batchUpsertPromises: Array<Promise<[PineconeUpsertResponse, NodePineconeVectorMap]>> = this.buildBatchedUpsertPromises(vectorBatches);

    const totalBatchUpsertResults = {
      upsertedNodeCount: 0,
      upsertedNodeIds: [] as string[],
      upsertedVectorCount: 0,
      upsertedVectorByNode: {} as Record<string, PineconeVector[]>,
      failedNodeCount: 0,
      failedNodeIds: [] as string[],
      errors: [] as string[]
    }
    // Wait for all batch upserts to complete, then sum the upserted counts.
    Promise.all(batchUpsertPromises)
      .then((successfulBatchUpserts) => {
        for (const [_upsertResponse, vectorsByNode] of successfulBatchUpserts) {
          for (const [nodeId, vectors] of Object.entries(vectorsByNode as NodePineconeVectorMap)) {
            totalBatchUpsertResults.upsertedNodeCount += 1;
            totalBatchUpsertResults.upsertedNodeIds.push(nodeId);
            totalBatchUpsertResults.upsertedVectorCount += vectors.length;
            totalBatchUpsertResults.upsertedVectorByNode[nodeId] ||= [];
            totalBatchUpsertResults.upsertedVectorByNode[nodeId].concat(vectors);
          }
        }
      }).catch((failedBatchUpserts) => {
        // For all of the failed upserts, add their node ids and collect their errors.
        for (const [error, vectorsByNode] of failedBatchUpserts) {
          for (const [nodeId, _vectors] of Object.entries(vectorsByNode as NodePineconeVectorMap)) {
            totalBatchUpsertResults.failedNodeCount += 1;
            totalBatchUpsertResults.failedNodeIds.push(nodeId);
            totalBatchUpsertResults.errors.push(error);
          }
        }
      });
    return totalBatchUpsertResults;
  }

  buildBatchedUpsertPromises(vectorBatches: Array<NodePineconeVectorMap>): Array<Promise<[PineconeUpsertResponse, NodePineconeVectorMap]>> {
    return vectorBatches.map(async (vectorBatch: NodePineconeVectorMap) => {
      // Accumulate all of the vectors in this batch into a single array.
      const allVectorsInBatch = Object.values(vectorBatch).reduce((allVectors, vectors) => {
        return allVectors.concat(vectors);
      }, [] as PineconeVector[]);

      // Upsert all of those vectors
      const batchUpsertResponse = this.pineconeIndex.upsert({
        upsertRequest: {
          vectors: allVectorsInBatch
        }
      }).then((upsertResponse: UpsertResponse) => {
        return Promise.resolve([upsertResponse, vectorBatch] as [PineconeUpsertResponse, NodePineconeVectorMap]);
      }).catch((e) => {
        return Promise.reject([`Error with call to Pinecone: ${e}`, vectorBatch])
      });
      return batchUpsertResponse;
    });
  }
}