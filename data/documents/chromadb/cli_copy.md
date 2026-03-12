# Copy Collections - Chroma Docs

> Source: https://docs.trychroma.com/docs/cli/copy

[Chroma Docshome page](/)
⌘
CLI
Copy Collections
[Docs](/docs/overview/introduction)
[Chroma Cloud](/cloud/getting-started)
[Guides](/guides/build/building-with-ai)
[Integrations](/integrations/chroma-integrations)
[Reference](/reference/overview)

##### Overview

- [Introduction](/docs/overview/introduction)
- [Getting Started](/docs/overview/getting-started)
- [Architecture](/docs/overview/architecture)
- [Open Source](/docs/overview/oss)
- [Migration](/docs/overview/migration)
- [Troubleshooting](/docs/overview/troubleshooting)

##### Run Chroma

- [Chroma Clients](/docs/run-chroma/clients)
- [Client-Server Mode](/docs/run-chroma/client-server)

##### Collections

- [Manage Collections](/docs/collections/manage-collections)
- [Add Data](/docs/collections/add-data)
- [Update Data](/docs/collections/update-data)
- [Delete Data](/docs/collections/delete-data)
- [Configure Collections](/docs/collections/configure)

##### Querying Collections

- [Query and Get](/docs/querying-collections/query-and-get)
- [Metadata Filtering](/docs/querying-collections/metadata-filtering)
- [Full Text Search](/docs/querying-collections/full-text-search)

##### Embeddings

- [Embedding Functions](/docs/embeddings/embedding-functions)
- [Multimodal Embeddings](/docs/embeddings/multimodal)

##### CLI

- [Installing the CLI](/docs/cli/install)
- [Browse Collections](/docs/cli/browse)
- [Copy Collections](/docs/cli/copy)
- [DB Management](/docs/cli/db)
- [Sample Apps](/docs/cli/sample-apps)
- [Login](/docs/cli/login)
- [Profile Management](/docs/cli/profile)
- [Run a Chroma Server](/docs/cli/run)
- [Update](/docs/cli/update)
- [Vacuum](/docs/cli/vacuum)

Using the Chroma CLI, you can copy collections from a local Chroma server to Chroma Cloud and vice versa.

```
chroma copy --from-local collections [collection names]

```

### ​Arguments

- `collections` - Space separated list of the names of the collections you want to copy. Conflicts with `all` .
- `all` - Instructs the CLI to copy all collections from the source DB.
- `from-local` - Sets the copy source to a local Chroma server. By default, the CLI will try to find it at `localhost:8000` . If you have a different setup, use `path` or `host` .
- `from-cloud` - Sets the copy source to a DB on Chroma Cloud.
- `to-local` - Sets the copy target to a local Chroma server. By default, the CLI will try to find it at `localhost:8000` . If you have a different setup, use `path` or `host` .
- `to-cloud` - Sets the copy target to a DB on Chroma Cloud.
- `db` - The name of the Chroma Cloud DB with the collections you want to copy. If not provided, the CLI will prompt you to select a DB from those available on your active [profile](./profile) .
- `host` - The host of your local Chroma server. This argument conflicts with `path` .
- `path` - The path of your local Chroma data. If provided, the CLI will use the data path to start a local Chroma server at an available port for browsing. This argument conflicts with `host` .

### ​Copy from Local to Chroma Cloud

simple
with DB
host
path

```
chroma copy --from-local collections col-1 col-2

```

### ​Copy from Chroma Cloud to Local

simple
with DB
host
path

```
chroma copy --from-cloud collections col-1 col-2

```

### ​Quotas

You may run into quota limitations when copying local collections to Chroma Cloud, for example if the size of your metadata values on records is too large. If the CLI notifies you that a quota has been exceeded, you can request an increase on the Chroma Cloud dashboard. Click “Settings” on your active profile’s team, and then choose the “Quotas” tab.

[Suggest edits](https://github.com/chroma-core/chroma/edit/main/docs/mintlify/docs/cli/copy.mdx)
[Browse CollectionsPrevious](/docs/cli/browse)
[DB ManagementNext](/docs/cli/db)
⌘
[Contact support](mailto:support@trychroma.com)