import { BaseNode } from "llamaindex";
import { NaiveSparseValuesBuilder, SparseValues, SparseValuesBuilderClass } from "./sparse_values";
import { SimpleMetadataBuilder, PineconeMetadataBuilderClass } from "..";
import { Vector } from "@pinecone-database/pinecone";

export type PineconeVectorsBuilderOptions = {
  alpha?: number;
  includeSparseValues?: boolean;
  dimension: number;
  splitEmbeddingsByDimension?: boolean;
  sparseVectorBuilder?: SparseValuesBuilderClass;
  pineconeMetadataBuilder?: PineconeMetadataBuilderClass;
  pineconeMetadataBuilderOptions?: Record<string, unknown>;
}

export class PineconeVectorsBuilder {
  node: BaseNode;
  embedding: number[];
  alpha: number = 0;
  includeSparseValues: boolean = false;
  splitEmbeddingsByDimension: boolean = false;
  dimension: number;
  sparseVectorBuilder: SparseValuesBuilderClass;
  pineconeMetadataBuilder: PineconeMetadataBuilderClass;
  pineconeMetadataBuilderOptions: Record<string, unknown>;

  constructor(node: BaseNode, embedding: number[], options: PineconeVectorsBuilderOptions) {
    this.node = node;
    this.embedding = this.normalizeEmbedding(embedding);

    this.alpha = options.alpha ?? 0;
    this.includeSparseValues = options.includeSparseValues ?? false;
    this.splitEmbeddingsByDimension = options.splitEmbeddingsByDimension ?? false;
    this.dimension = options.dimension;
    this.sparseVectorBuilder = options.sparseVectorBuilder ?? NaiveSparseValuesBuilder;
    this.pineconeMetadataBuilder = options.pineconeMetadataBuilder ?? SimpleMetadataBuilder;
    this.pineconeMetadataBuilderOptions = options.pineconeMetadataBuilderOptions ?? {};
  }

  // Some tokenizers return BigInts, which Pinecone doesn't like.
  private normalizeEmbedding(embedding: number[]): number[] {
    const numericEmbeddings = [];
    for (const embeddingValue of embedding) {
      numericEmbeddings.push(Number(embeddingValue).valueOf());
    }
    return numericEmbeddings;
  }

  public buildVectors(): Array<Vector> {
    let builtVectors: Array<Vector> = [];

    if (!this.splitEmbeddingsByDimension && this.embedding.length !== this.dimension) {
      throw new Error(`Node ${this.node.nodeId} has an embedding of length ${this.embedding.length}, but the index has a dimension of ${this.dimension}.`);
    }
    // If the embedding is less than or equal to the dimension,
    // build that one and move on. Its id will be the same as the node's nodeId.
    if (this.embedding.length <= this.dimension) {
      builtVectors = [this.buildVector(this.embedding)];

      // Otherwise, build multiple vectors, each with a unique id.
    } else {
      let vectorSubId = 0;
      for (const embeddingChunk of this.embeddingChunks()) {
        builtVectors.push(this.buildVector(embeddingChunk, vectorSubId++));
      }
    }
    return this.normalizedVectors(builtVectors);
  }

  private buildVector(embedding: number[], vectorSubId?: number): Vector {
    let vectorId;
    if (vectorSubId || vectorSubId === 0) {
      vectorId = `${this.node.nodeId}-${vectorSubId}`;
    } else {
      vectorId = this.node.nodeId;
    }

    const metadataBuilder = new this.pineconeMetadataBuilder(this.pineconeMetadataBuilderOptions);
    const metadata = metadataBuilder.buildMetadata(this.node);
    const vector: Vector = {
      id: vectorId,
      values: embedding,
      metadata
    };
    if (this.includeSparseValues) {
      vector.sparseValues = this.buildSparseValues(embedding);
    }
    return vector;
  }

  // Initializes a builder and outsources the work to its build method.
  private buildSparseValues(embedding: number[]): SparseValues {
    const builder = new this.sparseVectorBuilder(embedding);
    return builder.build();
  }

  // Generator to loop over embedding in chunks of dimension size.
  private *embeddingChunks(): Generator<Array<number>, void> {
    for (let chunkIndex = 0; chunkIndex < this.embedding.length; chunkIndex += this.dimension) {
      yield this.embedding.slice(chunkIndex, chunkIndex + this.dimension);
    }
  }

  // Pinecone requires that all vectors have the same dimension.
  // Because we split vectors by dimension length, only the final
  // sub-vector should be (possibly) too short.
  private normalizedVectors(vectors: Array<Vector>): Array<Vector> {
    const lastVector = vectors[vectors.length - 1];
    if (lastVector.values.length < this.dimension) {
      lastVector.values = lastVector.values.concat(Array(this.dimension - lastVector.values.length).fill(0));
    }
    return vectors;
  }

}