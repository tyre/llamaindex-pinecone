# LlamaIndex integration with Pinecone

This repository contains a LlamaIndex-compatible vector store backed by [Pinecone](https://pinecone.io) indices.

## Installation

`npm install llamaindex-pinecone`

To automatically have a working client initialized, site these environment variables:

- `PINECONE_API_KEY`: Your API Key 
- `PINECONE_ENVIRONMENT`: The environment of the pinecone instance you're using



## Usage

The heart and soul of this package is `PineconeVectorStore`. Let's see how that works.

### Creating a store

```typescript
import { PineconeVectorStore } from "llamaindex-pinecone";

// Initialize with the name of an index in Pinecone 
const vectorStore = new PineconeVectorStore({indexName: "speeches"});

// The pinecone client has to initialize asynchrously, so we need an extra
// call.
await vectorStore.init();
```
### Upserting vectors

Now let's do some things. Let's add some vectors.

```typescript
import { Document } from "llamaindex";

const aNormalNode = new Document({text: "Peter Piper picked a peck of pickled peppers. A peck of pickled peppers Peter Piper picked. If Peter Piper picked a peck of pickled peppers, Where's the peck of pickled peppers Peter Piper picked?", id_: "peter-piper"})
const embedding = [
  1, 5310, 7362, 546, 18691, 263, 1236, 384, 310, 5839, 839, 1236,
  22437, 29889, 319, 1236, 384, 310, 5839, 839, 1236, 22437, 5310, 7362,
  546, 18691, 29889, 960, 5310, 7362, 546, 18691, 263, 1236, 384, 310,
  5839, 839, 1236, 22437, 29892, 6804, 29915, 29879, 278, 1236, 384, 310,
  5839, 839, 1236, 22437, 5310, 7362, 546, 18691, 29973
];

const nodesWithEmbeddings = [{node: aNormalNode, embedding}];
await vectorStore.upsert(nodesWithEmbeddings);
// => ["peter-piper"], an array of vector ids upserted.
```

#### Sparse values

Upsertion supports sparse values, if `includeSparseValues: true` in passed in. By default, the sparse values will be built for you with a rather naive count-the-frequencies method.

With the same node and embedding above:

```typescript
await vectorStore.upsert(nodesWithEmbeddings, { includeSparseValues: true });
```
This will result in a sparse values dictionary included in vector upsert that looks like this:

```typescript
{
  indices: [
    1, 263, 278, 310, 319, 384, 546,
    839, 960, 1236, 5310, 5839, 6804, 7362,
    18691, 22437, 29879, 29889, 29892, 29915, 29973
  ],
  values: [
    1, 2, 1, 4, 1, 4, 4,
    4, 1, 8, 4, 4, 1, 4,
    4, 4, 1, 2, 1, 1, 1
  ]
}
```

#### How many vectors are created?

That depends on the dimensiality of your index.

If the embeddings passed into `upsert` are equal to the dimension of the index, there's nothing more to be done. You will have one vector.

If the embedding passed in is longer than the index, it will be split into multiple vectors. Here's a contrived example with a pinecone index whose dimension is `1`:

```typescript
const indexInfo = await myPineconeClient.Index("letters").describeIndexStats();
console.log(indexInfo.dimension)
// => 1

const node = TextNode({text: "word", id_:"wordNode"})
vectorStore.upsert([{ node , embedding: [23, 15, 18, 4]}])
// => ["wordNode-0", "wordNode-1", "wordNode-2", "wordNode-3"]
```

The API request to Pinecone would look something like this:

```JSON
{
  "vectors": [
    { "id": "wordNode-0", "values":[23], "metadata": { "nodeId": "wordNode" } },
    { "id": "wordNode-1", "values":[15], "metadata": { "nodeId": "wordNode" } },
    { "id": "wordNode-2", "values":[18], "metadata": { "nodeId": "wordNode" } },
    { "id": "wordNode-3", "values":[4], "metadata": { "nodeId": "wordNode" } }
  ]
}
```

Notice that the vector has been split to fit the dimension of the index. The ids in Pinecone have been adapted to be indexed in order, prefixed by the node's id, and the metadata is preserved.

The node's id is always included in the metadata, so deleting the document handles cleaning up all related vectors automatically. Query's can still filter by that node id in the metadata.

### Fetching vectors

Simple stuff. Note: this fetches vectors, not vectors for a node.
For nodes <= the dimension of the index, that's the same as the `node.nodeId`.

```typescript
vectorStore.client.fetch(["peter-piper"], "Namespace (Optional: defaults to default namespace)")
```

### Deleting vectors

#### By node id

Deletes all vectors associated with the given node ids.

🚨 NOTE 🚨
This does not work on Starter plans, which don't support filters on delete operations. Use `deleteVectors` instead.

```typescript
const nodeIds = ["wordNode", "peter-piper"];
await client.delete(nodeIds, "Namespace (Optional: defaults to default namespace)");
```

#### By vector id

```typescript
const vectorIds = [
  "wordNode-0",
  "wordNode-1"
  "wordNode-2",
  "wordNode-3"
];
await client.deleteVectors(vectorIds, "Namespace (Optional: defaults to default namespace)");
```

### Customization

`PineconeVectorStore` works well out of the box. You might want some customization, thought.

#### Customizing the client

By default, these Pinecone variables are pulled from the environment:

- `PINECONE_API_KEY`: Your API Key 
- `PINECONE_ENVIRONMENT`: The environment of the pinecone instance you're using

```typescript
import { PineconeClient } from "@pinecone-database/pinecone";

const myPineconeClient = new PineconeClient();
await myPineconeClient.init({ apiKey: "something secure", environment: "something environmental" });

const vectorStore = new PineconeVectorStore({ indexName: "UFO-files", client: myPineconeClient })
```

#### Customizing Sparse value generation

The naive sparse value generation is, as its name implies, naive. Other methods, like BM25 or SPLADE, may be more effective. The vector store supports passing a class that knows how to generate sparse values.

```typescript
import { SparseValues, SparseValueBuilder } from "llamaindex-pinecone";

class FancySparseValueBuilder implements SparseValueBuilder {
  embeddings: Array<number>;
  constructor(embeddings: Array<number>) {
    this.embeddings = embeddings;
  }

  build(): SparseValues {
    // Do fancy math to generate indices and values
    // Return a dictionary with those (aka `SparseValues`):
    return { indicies, values }
  }
}

const vectorStore = PineconeVectorStore("fancy-documents", {sparseValueBuilder: FancySparseValueBuilder});
```