import { BaseNode } from "llamaindex";
import { NaiveSparseValuesBuilder, SparseValues, SparseValuesBuilder } from "./sparse_values";
import { Vector } from "@pinecone-database/pinecone";

type PineconeMetadata = Record<string, string | number | boolean | Array<string>>;

type PineconeVectorsBuilderOptions = {
  includeSparseValues?: boolean;
  dimension: number;
  sparseVectorBuilder?: { new(embeddings: number[]): SparseValuesBuilder };
}

export class PineconeVectorsBuilder {
  node: BaseNode;
  embedding: number[];
  includeSparseValues: boolean;
  dimension: number;
  sparseVectorBuilder: { new(embeddings: number[]): SparseValuesBuilder };

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
    let vectorSubId = 0;
    const builtVectors: Array<Vector> = [];
    for (const embeddingChunk of this.embeddingChunks()) {
      builtVectors.push(this.buildVector(embeddingChunk, vectorSubId++));
    }

    return this.normalizedVectors(builtVectors);
  }

  private buildVector(embedding: number[], vectorSubId: number = 0): Vector {
    const vector: Vector = {
      id: `${this.node.nodeId}-${vectorSubId}`,
      values: embedding,
      metadata: this.extractNodeMetadata()
    };
    if (this.includeSparseValues) {
      vector.sparseValues = this.buildSparseValues(embedding);
    }
    return vector;
  }

  private buildSparseValues(embedding: number[]): SparseValues {
    const builder = new this.sparseVectorBuilder(embedding);
    return builder.build();
  }

  // Generator to loop over embedding
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
      id: this.node.nodeId,
      ...this.node.metadata
    };
  }

}