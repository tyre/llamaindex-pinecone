# LlamaIndex integration with Pinecone

This repository contains a LlamaIndex-compatible vector store backed by [Pinecone](https://pinecone.io) indices.

## Installation

~~npm install llamaindex-pinecone~~ NPM package outdated until deps update. Until then:

- Clone the typescript llamaindex package: `https://github.com/run-llama/LlamaIndexTS`
- Clone this package `git clone git@github.com:tyre/llamaindex-pinecone.git`
- In this repository, update `package.json` to include `"llamaindex":"file:path/to/LlamaIndexTS/packages/core"`
- In your project, update the `package.json` to include: `"llamaindex":"file:path/to/LlamaIndexTS/packages/core", "llamaindex-pinecone": "file:path/to/llamaindex-pinecone",`

At the end of the day, you need the edge version of the `llamaindex` package (from source) and update this package to point to it. Your package must point to the HEAD versions of each.

### API config

To automatically have a working client initialized, site these environment variables:

- `PINECONE_API_KEY`: Your API Key 
- `PINECONE_ENVIRONMENT`: The environment of the pinecone instance you're using


## Usage

The heart and soul of this package is `PineconeVectorStore`. Let's see how that works.

### `VectorStore` interface compliance

`PineconeVectorStore` adheres to the `VectorStore` interface from `llamaindex`:

```typescript
export interface VectorStore {
  storesText: boolean;
  isEmbeddingQuery?: boolean;
  client(): any;
  add(embeddingResults: NodeWithEmbedding[]): Promise<string[]>;
  delete(refDocId: string, deleteKwargs?: any): Promise<void>;
  query(query: VectorStoreQuery, kwargs?: any): Promise<VectorStoreQueryResult>;
  persist(persistPath: string, fs?: GenericFileSystem): Promise<void>;
}
```

### A basic integration

```typescript
import { storageContextFromDefaults, VectorStoreIndex } from "llamaindex";
import { FullContentMetadataBuilder, FullContentNodeHydrator, PineconeVectorStore } from "llamaindex-pinecone";

// Basic settings that have higher storage usage in Pinecone,
// but allow for quick plug-and-play
const easyModeSettings =   {
  // Store the entire node as JSON in the vector's metadata
  pineconeMetadataBuilder: FullContentMetadataBuilder,
  // When reading a vector, re-build the node from that JSON
  nodeHydrator: FullContentNodeHydrator 
}

// Initialize with the name of an index in Pinecone
const vectorStore = new PineconeVectorStore("speeches", easyModeSettings);

// define a storage context that's backed by our Pinecone vector store
const storageContext = await storageContextFromDefaults({ vectorStore })

// use that storage while we're loading documents
const vectorStoreIndex = await VectorStoreIndex.fromDocuments(presidentialInauguralAddresses, { storageContext });

// Make a query engine
const queryEngine = vectorStoreIndex.asQueryEngine();
// and ask it some questions!
const queryResponse = await queryEngine.query("What is the role of the Founding Fathers across inaugural addresses?");
```

And that's it!

Below is deeper documentation on interacting with the store itself. There is ample opportunity for customization.

## VectorQueryStore

### Adding a node to the store

Adding a node to the store requires us to pass a node and its embedding. The below example has already been tokenized:

```typescript
import { Document } from "llamaindex";

const vectorStore = new PineconeVectorStore("tongue-twisters");

const tongueTwister = new Document({text: "Peter Piper picked a peck of pickled peppers. A peck of pickled peppers Peter Piper picked. If Peter Piper picked a peck of pickled peppers, Where's the peck of pickled peppers Peter Piper picked?", id_: "peter-piper"})
const embedding = [
  1, 5310, 7362, 546, 18691, 263, 1236, 384, 310, 5839, 839, 1236,
  22437, 29889, 319, 1236, 384, 310, 5839, 839, 1236, 22437, 5310, 7362,
  546, 18691, 29889, 960, 5310, 7362, 546, 18691, 263, 1236, 384, 310,
  5839, 839, 1236, 22437, 29892, 6804, 29915, 29879, 278, 1236, 384, 310,
  5839, 839, 1236, 22437, 5310, 7362, 546, 18691, 29973
];
await vectorStore.add({ node: tongueTwister, emedding })
// => ["peter-piper]
```

We see that it has returned an array of node ids that were successfully upserted to Pinecone. Woohoo!

#### Batching

The call to `add` takes a set of options as its second argument. Calls to the Pinecone API are batched, in groups of 100 by default. Passing a `batchSize` changes this value:

```typescript
await vectorStore.add(nodesWithEmbeddings, { batchSize:  500 })
```

Changing the batch size possibly affects how many requests are made to Pinecone's API. You may want to fiddle with this if you are hitting rate limits. Note that Pinecone recommends batch sizes less than 1000.

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

### Querying

Now that we've put some data into Pinecone, let's take it back out again.

#### The old standby: `query`

Let's imagine a Pinecone index that contains vectors of the last 80 years of US President innaugural addresses. We'll now craft a query of "I love America"—a sentiment we'd expect to be pretty common!—and see the top five most relevant results:

```typescript
import { VectorStoreQueryMode } from "llamaindex";

const speechQuery = "I love America";
const queryEmbedding = Tokenizer.tokenize(speechQuery) // Tokenizer not included
const queryResponse = await vectorStore.query({ queryEmbedding, similarityTopK: 5, mode: VectorStoreQueryMode.DEFAULT });
// => {
//   nodes: [],
//   similarities: [ 0.134187013, 0.107230663, 0.095018886, 0.0797220916, 0.07660871 ],
//   ids: [
//     './speeches/inaugural-addresses/Dwight Eisenhower/1953.txt',
//     './speeches/inaugural-addresses/Ronald Reagan/1981.txt',
//     './speeches/inaugural-addresses/Franklin Delano Roosevelt/1945.txt',
//     './speeches/inaugural-addresses/Richard Nixon/1973.txt',
//     './speeches/inaugural-addresses/Harry Truman/1949.txt'
//   ]
// }
```

A top score of 0.13 is not inspiring, but, at least for README purposes, you can see the shape of the response. By default, the node ids are read from the `nodeId` field of the Pinecone vector's metadata. The similarities scores and ids correspond to each other by array index: the first similarity is for the first id, the second similarity for the second id, etc.

If you would like `query` to rebuild the nodes themselves, see "Hydrating nodes from the vector metadata" down below!

#### Looking for more: `queryAll`

This method is mostly analagous with `VectorStore.query` except that it returns Pinecone's matches rather than nodes.

```typescript
import { VectorStoreQuery, VectorStoreQueryMode } from 'llamaindex';

// A class that shows the subset of `VectorStoreQuery` fields supported
// by the PineconeVectorStore (and their API.)
class PineconeVectorStoreQuery implements VectorStoreQuery {
  queryEmbedding?: number[];
  similarityTopK: number;
  mode: VectorStoreQueryMode;
  alpha?: number;
  filters?: MetadataFilters;

  constructor(args: any) {
    Object.assign(this, args);
  }
}

const query = new PineconeVectorStoreQuery({
  similarityTopK: 3,
  queryEmbedding: [1,2,3,4,5],
  mode: VectorStoreQueryMode.SPARSE
});

await vectorStore.queryAll(query);
/** =>
 * {
 *   matches: [
 *     { id: "vector-1", score: 0.123 },
 *     { id: "vector-2", score: 0.32 },
*      { id: "vector-3", score: 0.119 }
 *   ]
 * }
```
#### Options

The second argument to `queryAll` is a dictionary of options. Those may include:

- `namespace`: the namespace to query. If not provided, will query the default namespace.
- `includeMetadata`: whether or not to include metadata in the response. Defaults to true.
- `includeValues`: whether or not to include matching vector values in the response. Defaults to true.
- `vectorId`: the id of a vector already in Pinecone that will be used as the query. Exclusive with `query.queryEmbedding`.

#### Return value

As we've seen, the response is an object with a key `matches` and an array of scored vectors. Each object contains:

- `id`: id of the vector
- `score`: the similarity to the query vector
- `values`: the vector, if `includeValues` was `true`
- `sparseValues`: the sparse values of the vector
- `metadata`: the metadata of the vector, if `includeMetadata` was `true`

### Fetching vectors

Simple stuff.

Note: this fetches vectors, not vectors for a node.
For nodes with an embedding <= the dimension of the index, that's the same as the `node.nodeId`.

```typescript
vectorStore.client.fetch(["peter-piper"], "Namespace (Optional: defaults to default namespace)")
// => {
//   namespace: "default",
//   vectors: {
//     "peter-piper": {
//       id: "peter-piper",
//       values: [1,2,3,3,4,5], // vector values
//       sparseValues: { indicies: [98, 412, 5, 12, 4], values: [3, 2, 1, 4, 2]},
//       metadata: {
//         nodeId: "peter-piper",
//         age: "old",
//         diffculty: "medium"
//       }
//     }
//   }
// }
```

The response contains a `vectors` object where the key is the vector id and the value is the vector.

### Deleting vectors

#### By node id

Deletes all vectors associated with the given node ids.

🚨 NOTE 🚨
This does not work on Free/Starter plans, which don't support filters on delete operations. Use `deleteVectors` instead.

```typescript
await client.delete("peter-piper");
```

#### For multiple nodes

```typescript
const nodeIds = ["wordNode", "peter-piper"];
await client.deleteAll(nodeIds);
```

#### By vector id

If you have the vectors' ids handy, you're welcome to delete those specifically:

```typescript
const vectorIds = [
  "wordNode-0",
  "wordNode-1"
  "wordNode-2",
  "wordNode-3"
];
await client.deleteVectors(vectorIds, "Namespace (Optional: defaults to default namespace)");
```

## Customization

`PineconeVectorStore` works well out of the box. You might want some customization, though.

### Customizing the client

By default, these Pinecone variables are pulled from the environment:

- `PINECONE_API_KEY`: Your API Key 
- `PINECONE_ENVIRONMENT`: The environment of the pinecone instance you're using

If you'd like to bring your own client, the `client` option will gladly accept yours.

```typescript
import { PineconeClient } from "@pinecone-database/pinecone";

const myPineconeClient = new PineconeClient();
await myPineconeClient.init({ apiKey: "something secure", environment: "something environmental" });

const vectorStore = new PineconeVectorStore({ indexName: "UFO-files", client: myPineconeClient })
```

### Customizing Sparse value generation

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

// Now we can use it!
const vectorStore = PineconeVectorStore("fancy-documents", { sparseValueBuilder: FancySparseValueBuilder });
```

When calling `add` or `upsert` with `includeSparseValues: true`, that builder will be used to generate sparse values being sent to Pinecone.

### Custom Vector Metadata Format

By default, PineconeVectorStore will upsert metadata in the form of:

```typescript
{
  nodeId: node.nodeId,
  ...node.metadata
}
```

To customize this, pass a class that implements `PineconeMetadataBuilder`. That means one methof of the signature `buildMetadata(node: BaseNode) => PineconeMetadata` as the `pineconeMetadataBuilder` option for the `PineconeVectorStore` constructor.

For example, if you want to include the full node's content serialized into the metadata, you might have:

```typescript
class FullContentMetadataBuilder implements PineconeMetadataBuilder {
  buildMetadata(node: BaseNode): PineconeMetadata {
    const nodeContent = JSON.stringify(node);
    const metadata: PineconeMetadata = {
      nodeId: node.id_,
      nodeType: node.getType(),
      nodeContent,
    };
    return metadata;
  }
}
const pineconeVectorStore = new PineconeVectorStore("speeches", { pineconeMetadataBuilder: FullContentMetadataBuilder });
await vectorStore.add({ node: tongueTwister, emedding });
```

In fact, this exact implementation is included out of the box. `import { FullContentMetadataBuilder } from "pinecone-llamaindex"` today!

### Custom Vector Metadata Options

Out of the box, `PineconeVectorStore.add` and `.upsert` use `SimpleMetadataBuilder`. It essentially adds the `nodeId` and then splats the `node.metadata` into an object. For Pinecone metadata, the only acceptable keys are string and values must be strings, numbers, booleans, or arrays of those types.

`pineconeMetadataBuilderOptions` is an optional parameter on the `PineconeVectorStore` constructor which will filter down to that builder. Fort the simple builder, the only presently supported option is `excludedMetadataKeys`, which takes an array of keys to skip.

```typescript
tongueTwister.metadata = {
  difficulty: "medium",
  age: "old",
  pineconeAPIKey: "xxxxxxxxxxxxxxxx"
}


const pineconeVectorStore = new PineconeVectorStore("speeches", { pineconeMetadataBuilderOptions });
const pineconeMetadataBuilderOptions = { excludedMetadataKeys: ["pineconeAPIKey"] }

await vectorStore.add({node: tongueTwister, embedding })
```

In this case, only `{ diffculty: "medium", age: "old" }` will be upserted to Pinecone, but the `tongueTwister` node itself will remain untouched.

This can be even more useful when implementing a custom `PineconeMetadataBuilder` as seen above.

### Hydrating nodes from the vector metadata

Returning the ids of nodes is fun, but maybe you'd like to re-build the nodes themselves. There are two options that can be passed into `PineconeVectorStore`'s constructor that enable this:

- `nodeHydrator`: a class conforming to NodeHydratorClass. Instances of this class must include a method of the signature `hydrate(vectorMetadata: PineconeMetadata) => BaseNode`;
- `nodeHydratorOptions`: Something you want passed to the constructor of the `nodeHydrator`

Included in this package is `FullContentNodeHydrator` which will:

- Look for a `nodeContent` property on the vector's metadata
- Parse it as JSON
- Look for a `nodeType`
- Match it against `ObjectType` in the llamaindex package
- If it is "DOCUMENT", "TEXT", or "INDEX", initialized that node type passing in the node content to its constructor.

To see how this works, less look at an example:

```typescript
import { Document } from "llamaindex";
import { FullContentNodeHydrator } from "llamaindex-pinecone";
// Example node object when it was originally upserted
const originalNodeData = { id_: "document-11", text: "secret data", metadata: { author: "CIA" } };
// What its vector metadata was upserted as
const pineconeVectorMetadata = {
  nodeId: "document-11",
  nodeType: "DOCUMENT",
  nodeContent: '{"id_":"document-11","text":"secret data","metadata":{"author":"CIA"}}'
};

const pineconeVectorStore = new PineconeVectorStore("speeches", { nodeHydrator: FullContentNodeHydrator })
const queryResults = await pineconeVectorStore.query(vectorStoreQuery);
const expectedDocument = new Document(originalNodeData);
expectedDocument == queryResults.nodes[0];
// => true
```

This pairs nicely with the example in "Custom Vector Metadata Format" above, which will upsert nodes in that format automatically.


## Advanced uses

### Upsert

Upsert is the underlying method that backs `add`. Let's look back at the example from `add`'s documentation above to see what else `upsert` tells us.


```typescript
import { Document } from "llamaindex";

const tongueTwister = new Document({text: "Peter Piper picked a peck of pickled peppers. A peck of pickled peppers Peter Piper picked. If Peter Piper picked a peck of pickled peppers, Where's the peck of pickled peppers Peter Piper picked?", id_: "peter-piper"})
const embedding = [
  1, 5310, 7362, 546, 18691, 263, 1236, 384, 310, 5839, 839, 1236,
  22437, 29889, 319, 1236, 384, 310, 5839, 839, 1236, 22437, 5310, 7362,
  546, 18691, 29889, 960, 5310, 7362, 546, 18691, 263, 1236, 384, 310,
  5839, 839, 1236, 22437, 29892, 6804, 29915, 29879, 278, 1236, 384, 310,
  5839, 839, 1236, 22437, 5310, 7362, 546, 18691, 29973
];

const nodesWithEmbeddings = [{node: tongueTwister, embedding}];
await vectorStore.upsert(nodesWithEmbeddings);
/**
 * => {
 *   upsertedNodeCount: 1,
 *   upsertedNodeIds: ["peter-piper"],
 *   upsertedVectorCount: 1,
 *   upsertedVectorByNode: { "peter-piper": ["peter-piper"] }
 *   failedNodeCount: 0,
 *   failedNodeIds: [],
 *   errors: []
 *  }
*/
```

It's a lot! Let's break down what's in there:

- `upsertedNodeCount`: the total number of nodes upserted,
- `upsertedNodeIds`: the node ids that were successfully upserted
- `upsertedVectorCount`: the total number of vectors upserted
- `upsertedVectorByNode`: a mapping of the node ids to the vectors that were upserted for that node
- `failedNodeCount`: the number of nodes that failed to *fully* upsert
- `failedNodeIds`: the ids of the nodes that failed to *fully* upsert
- `errors`: an array of errors that occurred during upsert

For most cases, `upsertedNodeCount` and `upsertedVectorCount` will be the same. See  the "Automatic Embedding Splitting" section below for the option to split nodes across multiple vectors.

#### Passing duplicate vectors

Note that the passing duplicate nodes—those with identical node ids—and embeddings will only create one vector in Pinecone, but the response will count both. The returned array will return the nodeId twice.


#### Automatic Embedding Splitting

By default, the embedding for a node is expected to be exactly as long as the dimension of the index. If not, `PineconeVectorStore` will throw an error.

If you have a large embedding that you would like automatically, though naively, chunked into multiple vectors the size of the index's dimension, set `splitEmbeddingsByDimension` to `true`. This is most useful for testing.

Here's a contrived example with a pinecone index whose dimension is `1`:

```typescript
const indexInfo = await myPineconeClient.Index("letters").describeIndexStats();
console.log(indexInfo.dimension)
// => 1

const node = TextNode({text: "word", id_:"wordNode"})
vectorStore.upsert([{ node , embedding: [23, 15, 18, 4]}], { splitEmbeddingsByDimension: true })
/**
 * => {
 *   upsertedNodeCount: 1,
 *   upsertedNodeIds: ["wordNode"],
 *   upsertedVectorCount: 4,
 *   upsertedVectorByNode: {
 *     "wordNode": [
 *       "wordNode-0",
 *       "wordNode-1",
 *       "wordNode-2",
 *       "wordNode-3"
 *     ]
 *   },
 *   failedNodeCount: 0,
 *   failedNodeIds: [],
 *   errors: []
 *  }
*/
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

##### With batching

Note that for batched requests, the vectors for a given node can be spread across multiple requests. These requests can succeed or fail independently, so it is possible for a given node id to be in both the upserted and failed lists. Since these are upserts, it is safe to retry the failed nodes.