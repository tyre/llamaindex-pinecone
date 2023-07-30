import { NodeWithEmbedding } from "llamaindex";
import { PineconeClient, Vector } from "@pinecone-database/pinecone";
import { SparseValuesBuilder, NaiveSparseValuesBuilder, utils, PineconeVectorsBuilder } from ".";
import { DeleteRequest, VectorOperationsApi } from "@pinecone-database/pinecone/dist/pinecone-generated-ts-fetch";
import { PineconeUpsertOptions, PineconeUpsertResults, PineconeVectorsUpsert } from "./pinecone_api";

type PineconeVectorStoreOptions = {
  indexName: string;
  client?: PineconeClient;
  namespace?: string;
  // class that implements SparseValuesBuilder.
  // Can't to `typeof` with an interface :(
  sparseVectorBuilder?: { new(embeddings: number[]): SparseValuesBuilder };
}

type PineconeIndex = VectorOperationsApi;

type PineconeNamespaceSummary = {
  vectorCount?: number;
}

type PineconeIndexStats = {
  // If there are namespaces for this index,
  // this will be a map of namespace name to a summary for
  // that namespace. At the moment, that only includes the count
  // of the vectors.
  namespaces?: Record<string, PineconeNamespaceSummary>;
  dimension: number;
  indexFullness: number;
  totalVectorCount: number;
}

export class PineconeVectorStore {
  indexName: string;
  namespace: string | undefined;
  client: PineconeClient | undefined;
  sparseVectorBuilder: { new(embeddings: number[]): SparseValuesBuilder };

  pineconeIndex: PineconeIndex | undefined;
  pineconeIndexStats: PineconeIndexStats | undefined;

  constructor(options: PineconeVectorStoreOptions) {
    this.client = options.client;
    this.indexName = options.indexName;
    this.namespace = options.namespace;
    this.sparseVectorBuilder = options.sparseVectorBuilder || NaiveSparseValuesBuilder;
  }

  /**
   * Returns or initializes a PineconeClient instance. This is necessary
   * because the Pinecone client requires an async init function, so it
   * cannot be initialized in the constructor.
   *
   * @returns {Promise<PineconeClient>} PineconeClient instance
   */
  async getPineconeClient(): Promise<PineconeClient> {
    if (this.client) {
      return Promise.resolve(this.client);
    }
    this.client = new PineconeClient();
    await this.client.init(utils.getPineconeConfigFromEnv());
    return this.client;
  }

  /**
   * Returns a pinecone index instance for the store's indexName.
   *  
   * An Index is actually an API clientish object (VectorOperationsApi)
   * that allows operations on indicies (e.g. upsert, querying, etc.)
   * backed by the API. This is aliased to the PineconeIndex type.
   * 
   * @returns {Promise<PineconeIndex>} Pinecone Index instance
   */
  async getIndex(forceRefresh: boolean = false): Promise<PineconeIndex> {
    if (this.pineconeIndex && !forceRefresh) {
      return Promise.resolve(this.pineconeIndex);
    }
    const client = await this.getPineconeClient();
    this.pineconeIndex = client.Index(this.indexName);
    return this.pineconeIndex;
  }

  async getIndexStats(forceRefresh: boolean = false): Promise<PineconeIndexStats> {
    if (this.pineconeIndexStats && !forceRefresh) {
      return Promise.resolve(this.pineconeIndexStats);
    }
    const index = await this.getIndex();
    const indexStats = await index.describeIndexStats({ describeIndexStatsRequest: {} });
    this.pineconeIndexStats = indexStats as PineconeIndexStats;
    return Promise.resolve(this.pineconeIndexStats);
  }

  /**
   * Upsert llamaindex Nodes into a Pinecone index.
   * Builds vectors from the embeddings of the nodes, including metadata,
   * then upserts them into the index.
   * Optionally includes sparse values and pads embeddings to the index's dimension.
   * 
   * @param {NodeWithEmbedding[]} nodesWithEmbeddings - an array of Nodes with embeddings.
   * @param {PineconeUpsertOptions} upsertOptions - options for the upsert operation, like
   *    batch size and whether to include sparse values.
   * @returns {PineconeUpsertResults} - an object that includes the number of successful
   *    upserts, the ids of vectors upserted, the number of failed upserts, and the ids
   *    for vectors that failed to upsert.
   * 
   *    Note that for batched requests, if the response for the batch indicates that fewer
   *    vectors were affected than were in the batch, that batch is considered failed. Since
   *    we are upserting, it's safe to retry all nodes/embeddings from that entire batch.
  */
  async upsert(
    nodesWithEmbeddings: NodeWithEmbedding[],
    upsertOptions: PineconeUpsertOptions = {}
  ): Promise<PineconeUpsertResults> {
    const builtVectors: Array<Vector> = [];
    // Fetch stats about the index so we know its dimension.
    const indexStats = await this.getIndexStats();

    // Loop over each of the nodes with embeddings, build vectors. We flatten
    // all vectors into a single list for upsertion.
    for (const nodeWithEmbedding of nodesWithEmbeddings) {
      const node = nodeWithEmbedding.node;
      const embedding = nodeWithEmbedding.embedding;
      if (!embedding) {
        throw new Error(`Node ${node.nodeId} does not have an embedding.`);
      }

      // Build the vectors for this node + embedding pair.
      const vectorsBuilder = new PineconeVectorsBuilder(
        node,
        embedding,
        { includeSparseValues: upsertOptions.includeSparseValues, dimension: indexStats.dimension }
      );
      const vectors = vectorsBuilder.buildVectors()
      builtVectors.push(...vectors);
    }

    const vectorsUpsert = new PineconeVectorsUpsert(await this.getIndex(), upsertOptions)
    return vectorsUpsert.execute(builtVectors);
  }

  async fetch(pineconeVectorIds: string[], namespace?: string): Promise<Record<string, Vector>> {
    const index = await this.getIndex();
    const fetchResponse = await index.fetch({ ids: pineconeVectorIds, namespace });
    return fetchResponse.vectors || {};
  }

  /**
   * Delete vectors from the index by vector id.
   * @param {string[]} pineconeVectorIds - an array of vector ids to delete.
   * @param {string} namespace - the namespace to delete from. If not provided, will delete from the default namespace.
   * @returns {Promise<object>} the response from the delete call. No documentation exists for this response, but it's likely empty.
   */
  async deleteVectors(pineconeVectorIds: string[], namespace?: string): Promise<object> {
    const index = await this.getIndex();
    const requestParams: Partial<DeleteRequest> = { ids: pineconeVectorIds };
    if (namespace) {
      requestParams.namespace = namespace;
    }

    const deleteResponse = await index._delete({
      deleteRequest: requestParams
    });
    return deleteResponse
  }

  /**
   * Delete vectors from the index by nodeId.
   * 
   * @param {string[]} nodeIds - an array of nodeIds to delete.
   * @param {string} namespace - the namespace to delete from.
   *    If not provided, will delete from the default namespace.
   * @returns {Promise<object>} - the response from the Pinecone API call.
   *    It is undocumented and seems to always be an empty object,
   *    but you might wish to check for yourself.
   * 
   * ***************
   * ** IMPORTANT **
   * ***************
   * 
   * This does not work on Starter plans, as they do not allow
   * filters in deletion requests.
   */
  async delete(nodeIds: string[], namespace?: string): Promise<object> {
    const index = await this.getIndex();
    const requestParams: Partial<DeleteRequest> = {
      filter: { "nodeId": { "$in": nodeIds } }
    };
    if (namespace) {
      requestParams.namespace = namespace;
    }
    const deleteResponse = await index._delete({
      deleteRequest: requestParams
    });

    return deleteResponse;
  }

}
