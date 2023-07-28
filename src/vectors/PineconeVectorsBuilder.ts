import { BaseNode } from "llamaindex";
import { NaiveSparseValuesBuilder, SparseValues, SparseValuesBuilder } from "./sparse_values";
import { Vector } from "@pinecone-database/pinecone";

type PineconeMetadata = Record<string, string | number | boolean | Array<string>>;
type SparseValuesBuilderClass = { new(embeddings: number[]): SparseValuesBuilder };

type PineconeVectorsBuilderOptions = {
  includeSparseValues?: boolean;
  dimension: number;
  sparseVectorBuilder?: SparseValuesBuilderClass;
}

export class PineconeVectorsBuilder {
  node: BaseNode;
  embedding: number[];
  includeSparseValues: boolean;
  dimension: number;
  sparseVectorBuilder: SparseValuesBuilderClass = NaiveSparseValuesBuilder;

  constructor(node: BaseNode, embedding: number[], options: PineconeVectorsBuilderOptions) {
    this.node = node;
    this.embedding = this.normalizeEmbedding(embedding);
    this.includeSparseValues = options.includeSparseValues || false;
    this.dimension = options.dimension;
    this.sparseVectorBuilder = options.sparseVectorBuilder || NaiveSparseValuesBuilder;
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
    const vector: Vector = {
      id: vectorId,
      values: embedding,
      metadata: this.extractNodeMetadata()
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
  private normalizedVectors(vectors: Array<Vector>): Array<Vector> {
    const lastVector = vectors[vectors.length - 1];
    if (lastVector.values.length < this.dimension) {
      lastVector.values = lastVector.values.concat(Array(this.dimension - lastVector.values.length).fill(0));
    }
    return vectors;
  }

  private extractNodeMetadata(): PineconeMetadata {
    return {
      nodeId: this.node.nodeId,
      ...this.node.metadata
    };
  }
}