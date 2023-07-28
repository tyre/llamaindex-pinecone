import { NodeWithEmbedding, BaseNode } from "llamaindex";
import { PineconeClient, Vector } from "@pinecone-database/pinecone";
import { NaiveSparseValueBuilder, SparseValueBuilder, SparseValues, utils } from ".";
import { VectorOperationsApi } from "@pinecone-database/pinecone/dist/pinecone-generated-ts-fetch";

type PineconeVectorStoreOptions = {
  indexName: string;
  pineconeClient?: PineconeClient;
  namespace?: string;
  // class that implements SparseValueBuilder.
  // Can't to `typeof` with an interface :(
  sparseVectorBuilder?: { new(embeddings: number[]): SparseValueBuilder };
}

type PineconeIndex = VectorOperationsApi;

type PineconeUpsertOptions = {
  batchSize?: number;
  includeSparseValues?: boolean;
}

export class PineconeVectorStore {
  indexName: string;
  namespace: string | undefined;
  pineconeClient: PineconeClient | undefined;
  sparseVectorBuilder: { new(embeddings: number[]): SparseValueBuilder };

  pineconeIndex: PineconeIndex | undefined;

  constructor(options: PineconeVectorStoreOptions) {
    this.pineconeClient = options.pineconeClient;
    this.indexName = options.indexName;
    this.namespace = options.namespace;
    this.sparseVectorBuilder = options.sparseVectorBuilder || NaiveSparseValueBuilder;
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
   * Returns a pinecone index instance for the store's indexName.
   *  
   * An Index is actually an API clientish object (VectorOperationsApi)
   * that allows operations on indicies (e.g. upsert, querying, etc.)
   * backed by the API. This is aliased to the PineconeIndex type.
   * 
   * @returns {Promise<PineconeIndex>} Pinecone Index instance
   */
  async getIndex(): Promise<PineconeIndex> {
    if (this.pineconeIndex) {
      return Promise.resolve(this.pineconeIndex);
    }
    const pineconeClient = await this.getPineconeClient();
    this.pineconeIndex = pineconeClient.Index(this.indexName);
    return this.pineconeIndex;
  }

  /**
   * Upsert llamaindex Nodes into a Pinecone index.
   * @param {NodeWithEmbedding[]} nodesWithEmbeddings - an array of Nodes with embeddings.
   * @returns an array of Pinecone IDs that were created or updated.
  */
  async upsert(
    nodesWithEmbeddings: NodeWithEmbedding[],
    upsertOptions: PineconeUpsertOptions = {}
  ): Promise<number[]> {
    const pineconeClient = await this.getPineconeClient();

    for (const nodeWithEmbedding of nodesWithEmbeddings) {
      const node = nodeWithEmbedding.node;
      const embedding = nodeWithEmbedding.embedding;
      if (!embedding) {
        throw new Error(`Node ${node.nodeId} does not have an embedding.`);
      }
      const vector = this.buildVector(node, embedding, upsertOptions.includeSparseValues || false);
      console.log(vector);

      // await pineconeClient.upsert(this.indexName, vector);
    }

    return [];
  }

  buildVector(node: BaseNode, embedding: number[], includeSparseValues: Boolean): Vector {
    const vector: Vector = {
      id: node.nodeId,
      values: embedding!,
      metadata: extractNodeMetadata(node)
    };
    if (includeSparseValues) {
      vector.sparseValues = this.buildSparseValues(embedding);
    }
    return vector;
  }

  buildSparseValues(embedding: number[]): SparseValues {
    const builder = new this.sparseVectorBuilder(embedding);
    return builder.build();
  }
}

export function extractNodeMetadata(node: BaseNode): Record<string, any> {
  return {
    id: node.nodeId,
  };
}


// def node_to_metadata_dict(
//   node: BaseNode,
//   remove_text: bool = False,
//   text_field: str = DEFAULT_TEXT_KEY,
//   flat_metadata: bool = False,
// ) -> dict:
//   """Common logic for saving Node data into metadata dict."""
//   metadata: Dict[str, Any] = node.metadata

//   if flat_metadata:
//       _validate_is_flat_dict(metadata)

//   # store entire node as json string - some minor text duplication
//   node_dict = node.dict()
//   if remove_text:
//       node_dict[text_field] = ""

//   # remove embedding from node_dict
//   node_dict["embedding"] = None

//   # dump remainer of node_dict to json string
//   metadata["_node_content"] = json.dumps(node_dict)

//   # store ref doc id at top level to allow metadata filtering
//   # kept for backwards compatibility, will consolidate in future
//   metadata["document_id"] = node.ref_doc_id or "None"  # for Chroma
//   metadata["doc_id"] = node.ref_doc_id or "None"  # for Pinecone, Qdrant, Redis
//   metadata["ref_doc_id"] = node.ref_doc_id or "None"  # for Weaviate

//   return metadata

// export interface Vector {
//     /**
//      * This is the vector's unique id.
//      * @type {string}
//      * @memberof Vector
//      */
//     id: string;
//     /**
//      * This is the vector data included in the request.
//      * @type {Array<number>}
//      * @memberof Vector
//      */
//     values: Array<number>;
//     /**
//      * 
//      * @type {SparseValues}
//      * @memberof Vector
//      */
//     sparseValues?: SparseValues;
//     /**
//      * This is the metadata included in the request.
//      * @type {object}
//      * @memberof Vector
//      */
//     metadata?: object;
// }