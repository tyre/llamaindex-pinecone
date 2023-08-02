import {
  NodeWithEmbedding, BaseNode, GenericFileSystem,
  VectorStoreQueryResult,
  VectorStoreQueryMode,
  MetadataFilters,
  VectorStoreQuery,
  VectorStore
} from "llamaindex";
import { PineconeClient, Vector as PineconeVector, ScoredVector as PineconeScoredVector } from "@pinecone-database/pinecone";
import { SparseValuesBuilder, NaiveSparseValuesBuilder, utils, PineconeVectorsBuilder, PineconeVectorsBuilderOptions } from ".";
import { DeleteRequest, VectorOperationsApi as PineconeIndex } from "@pinecone-database/pinecone/dist/pinecone-generated-ts-fetch";
import { PineconeQueryBuilder, PineconeUpsertOptions, PineconeUpsertResults, PineconeVectorsUpsert, PineconeQueryBuilderOptions, PineconeUpsertVectorsRecord } from "./pinecone_api";

type PineconeVectorStoreOptions = {
  indexName: string;
  pineconeClient?: PineconeClient;
  namespace?: string;
  // class that implements SparseValuesBuilder.
  // Can't to `typeof` with an interface :(
  sparseVectorBuilder?: { new(embeddings: number[]): SparseValuesBuilder };
}

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

export class PineconeVectorStore implements VectorStore {
  storesText: boolean = true;
  indexName: string;
  namespace: string | undefined;
  pineconeClient: PineconeClient | undefined;
  sparseVectorBuilder: { new(embeddings: number[]): SparseValuesBuilder };

  pineconeIndex: PineconeIndex | undefined;
  pineconeIndexStats: PineconeIndexStats | undefined;

  constructor(options: PineconeVectorStoreOptions) {
    this.pineconeClient = options.pineconeClient;
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
    if (this.pineconeClient) {
      return Promise.resolve(this.pineconeClient);
    }
    this.pineconeClient = new PineconeClient();
    await this.pineconeClient.init(utils.getPineconeConfigFromEnv());
    return this.pineconeClient;
  }


  /**
   * Returns the underlying pinecone client.
   * Note: may need to be initialized, which can be guaranteed by either:
   * 1. Passing an initialized client via the `pineconeClient` option in the constructor
   * 2. Calling `getPineconeClient` when no client is passed in.
   * @returns {PineconeClient}
   */
  get client(): any {
    return this.pineconeClient;
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
   * 
   * @param embeddingResults - A list of nodes with their embeddings
   * @param kwargs - Keyword arguments for the upsert operation
   * @returns List of node ids that had their embeddings uploaded
   * 
   * This is a simplified call `upsert` behind the scenes.
   */
  async add(embeddingResults: NodeWithEmbedding[], kwargs?: any): Promise<string[]> {
    const upsertResults = await this.upsert(embeddingResults, kwargs);
    return upsertResults.upsertedNodeIds;
  }

  // The query signature for VectorStore requires returning a VectorStoreQueryResult,
  // which includes nodes. The only way to return nodes is to rebuild them from a pinecone
  // query result. Because nodes contain the source data (e.g. the text), we would have to
  // store the entire text in the metadata of the vector. Under discussion whether that's a
  // good idea and, if not, whether the library should encourage it.
  async query(query: VectorStoreQuery, kwargs?: any): Promise<VectorStoreQueryResult> {
    throw "Unimplemented. Users should instead use `queryAll` to return an array of scored vectors.\
    These can then be matched with the original nodes via `vector.metdata.nodeId`."
    return Promise.resolve({ ids: [], similarities: [] });
  }

  /**
   * Query the index for vectors similar to the query vector.
   * @param {VectorStoreQuery} query - the query to execute.
   * @param {any} kwargs - keyword arguments for the query. Supports:
   *  - namespace: the namespace to query. If not provided, will query the default namespace.
   *  - includeMetadata: whether or not to include metadata in the response. Defaults to true.
   *  - includeValues: whether or not to include matching vector values in the response. Defaults to true.
   *  - vectorId: the id of a vector already in Pinecone that will be used as the query.
   *    Exclusive with `query.queryEmbedding`.
  
   * @returns {Promise<Array<PineconeScoredVector>>} - an array of scored vectors of length <= topK.
   *   Each object contains:
   *    - id: id of the vector
   *    - score: the similarity to the query vector
   *    - values: the vector, if includeValues was true
   *    - sparseValues: the sparse values of the vector
   *    - metadata: the metadata of the vector, if includeMetadata was true
  */
  async queryAll(query: VectorStoreQuery, kwargs?: any): Promise<Array<PineconeScoredVector>> {
    const passedArguments = Object.keys(kwargs || {});
    const queryBuilderOptions: PineconeQueryBuilderOptions = {
      vector: query.queryEmbedding,
      topK: query.similarityTopK,
      alpha: query.alpha,
      // @ts-ignore
      filters: query.filters,
      namespace: kwargs?.namespace,
      id: kwargs?.vectorId,
    }

    if (passedArguments.includes("includeMetadata"))
      queryBuilderOptions.includeMetadata = kwargs.includeMetadata;
    if (passedArguments.includes("includeValues"))
      queryBuilderOptions.includeValues = kwargs.includeValues;

    if (query.mode === VectorStoreQueryMode.SPARSE || query.mode === VectorStoreQueryMode.HYBRID) {
      const vectorBuilderClass = this.sparseVectorBuilder;
      const vectorBuilder: SparseValuesBuilder = new vectorBuilderClass(query.queryEmbedding!);
      queryBuilderOptions.sparseVector = vectorBuilder.build();
    }

    const queryBuilder = new PineconeQueryBuilder(queryBuilderOptions);

    const index = await this.getIndex()
    const { matches } = await index.query({ queryRequest: queryBuilder.toQueryRequest() });
    return Promise.resolve(matches || []);
  }


  DEFAULT_UPSERT_OPTIONS = {
    splitEmbeddingsByDimension: false,
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
   * @returns {PineconeUpsertResults} - an object that includes:
   * 
   *  - upsertedNodeCount: the total number of nodes upserted,
   *  - upsertedNodeIds: the node ids that were successfully upserted
   *  - upsertedVectorCount: the total number of vectors upserted
   *  - upsertedVectorByNode: a mapping of the node ids to the vectors that were upserted for that node
   *  - failedNodeCount: the number of nodes that failed to *fully* upsert
   *  - failedNodeIds: the ids of the nodes that failed to *fully* upsert
   *  - errors: an array of errors that occurred during upsert
   *   
   * 
   *  Note that for batched requests, the vectors for a given node can be spread
   *  across multiple requests. These requests can succeed or fail independently, so 
   *  it is possible for a given node id to be in both the upserted and failed lists.
   *  Since these are upserts, it is safe to retry the failed nodes.
   * 
  */
  async upsert(
    nodesWithEmbeddings: NodeWithEmbedding[],
    upsertOptions: PineconeUpsertOptions = this.DEFAULT_UPSERT_OPTIONS
  ): Promise<PineconeUpsertResults> {
    const builtVectorsByNode: PineconeUpsertVectorsRecord = { totalVectorCount: 0, vectorsByNode: {} };
    // Fetch stats about the index so we know its dimension.
    const indexStats = await this.getIndexStats();

    // Loop over each of the nodes with embeddings, build vectors. We keep
    // a dictionary of {node => [vectors]} to track which vectors belong to which nodes
    for (const nodeWithEmbedding of nodesWithEmbeddings) {
      const node = nodeWithEmbedding.node;
      const embedding = nodeWithEmbedding.embedding;
      if (!embedding) {
        throw new Error(`Node ${node.nodeId} does not have an embedding.`);
      }

      const vectorBuilderOptions: PineconeVectorsBuilderOptions = {
        includeSparseValues: upsertOptions.includeSparseValues,
        dimension: indexStats.dimension,
        splitEmbeddingsByDimension: upsertOptions.splitEmbeddingsByDimension,
        sparseVectorBuilder: this.sparseVectorBuilder
      }

      if (upsertOptions.extractPineconeMetadata)
        vectorBuilderOptions.extractPineconeMetadata = upsertOptions.extractPineconeMetadata;

      // Build the vectors for this node + embedding pair.
      const vectorsBuilder = new PineconeVectorsBuilder(
        node,
        embedding,
        vectorBuilderOptions
      );
      const vectors = vectorsBuilder.buildVectors()
      builtVectorsByNode.totalVectorCount += vectors.length;
      builtVectorsByNode.vectorsByNode[node.nodeId] = vectors;
    }

    const vectorsUpsert = new PineconeVectorsUpsert(await this.getIndex(), upsertOptions)
    return vectorsUpsert.execute(builtVectorsByNode);
  }

  async fetch(pineconeVectorIds: string[], namespace?: string): Promise<Record<string, PineconeVector>> {
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
  async deleteAll(nodeIds: string[], namespace?: string): Promise<object> {
    const index = await this.getIndex();
    const requestParams: Partial<DeleteRequest> = {
      filter: { "nodeId": { "$in": nodeIds } }
    };
    if (namespace) requestParams.namespace = namespace;
    const deleteResponse = await index._delete({
      deleteRequest: requestParams
    });

    return deleteResponse;
  }

  async delete(refDocId: string, deleteKwargs?: any): Promise<void> {
    await this.deleteAll([refDocId], deleteKwargs?.namespace);
    return Promise.resolve();
  }

  // No-op implementation to fulfill VectorStore interface.
  // We've been persisting all along!
  async persist(_persistPath: string, _fs?: GenericFileSystem): Promise<void> {
    return;
  }

}
