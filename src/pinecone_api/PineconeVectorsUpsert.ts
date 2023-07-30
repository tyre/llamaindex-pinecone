import { Vector } from "@pinecone-database/pinecone";
import { PineconeUpsertOptions, PineconeUpsertResults } from "./types";
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

  async execute(pineconeVectors: Vector[]) {
    const batchSize = this.batchSize;
    if (pineconeVectors.length > (batchSize)) {
      return this.batchUpsert(pineconeVectors, batchSize);
    } else {
      return this.singleUpsert(pineconeVectors);
    }
  }

  // Upserts a single array of pinecone vectors. This assumes that the vectors are
  // less than or equal to the desired batch size.
  async singleUpsert(builtVectors: Vector[]): Promise<PineconeUpsertResults> {
    let upsertedCount = 0;
    try {
      const upsertResponse = await this.pineconeIndex.upsert({ upsertRequest: { vectors: builtVectors } });
      upsertedCount = upsertResponse.upsertedCount || 0;
    } catch (e) {
      throw `Error with call to Pinecone: ${e}`;
    }

    const vectorIds = builtVectors.map((vector) => vector.id);
    console.log({ upsertedCount });
    const upsertResults = {
      upsertedCount,
      failedCount: 0,
      upsertedVectorIds:
        vectorIds,
      failedVectorIds: []
    };
    return Promise.resolve(upsertResults);
  }

  vectorBatches(vectors: Vector[], batchSize: number): Array<Array<Vector>> {
    const vectorBatches: Array<Array<Vector>> = [];
    for (let i = 0; i < vectors.length; i += batchSize) {
      vectorBatches.push(vectors.slice(i, i + batchSize));
    }
    return vectorBatches;
  }

  /*
  * Batch upsert vectors into the index.
  * @param {Vector[]} builtVectors - an array of vectors to upsert.
  * @param {number} batchSize - the size of each batch to upsert.
  * @returns {Promise<number>} the number of vectors affected.
  */
  async batchUpsert(builtVectors: Vector[], batchSize: number): Promise<PineconeUpsertResults> {
    const vectorBatches = this.vectorBatches(builtVectors, batchSize);
    const batchUpsertPromises: Array<Promise<[PineconeUpsertResponse, Vector[]]>> = vectorBatches.map(async (vectorBatch: Array<Vector>) => {
      const batchUpsertResponse = this.pineconeIndex.upsert({
        upsertRequest: {
          vectors: vectorBatch
        }
      }).then((upsertResponse: UpsertResponse) => {
        return Promise.resolve([upsertResponse, vectorBatch] as [PineconeUpsertResponse, Vector[]]);
      });
      return batchUpsertResponse;
    });

    const emptyBatchUpsertResults = {
      upsertedCount: 0,
      upsertedVectorIds: [] as string[],
      failedCount: 0,
      failedVectorIds: [] as string[]
    }
    // Wait for all batch upserts to complete, then sum the upserted counts.
    const batchUpsertResponses = await Promise.all(batchUpsertPromises);
    return batchUpsertResponses.reduce((upsertResults, [upsertResponse, vectorBatch]): PineconeUpsertResults => {
      const vectorBatchIds = vectorBatch.map((vector) => vector.id);
      if (upsertResponse.upsertedCount == vectorBatch.length) {
        upsertResults.upsertedCount = upsertResponse.upsertedCount! + upsertResults.upsertedCount;
        upsertResults.upsertedVectorIds = upsertResults.upsertedVectorIds.concat(vectorBatchIds)
        return upsertResults;
      }
      return upsertResults;
    }, emptyBatchUpsertResults);
  }
}