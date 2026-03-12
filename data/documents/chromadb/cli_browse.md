# Browse Collections - Chroma Docs

> Source: https://docs.trychroma.com/docs/cli/browse

[Chroma Docshome page](/)
⌘
CLI
Browse Collections
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

You can use the Chroma CLI to inspect your collections with an in-terminal UI. The CLI supports browsing collections from DBs on Chroma Cloud or a local Chroma server.

```
chroma browse [collection_name] [--local]

```

### ​Arguments

- `collection_name` - The name of the collection you want to browse. This is a required argument.
- `db_name` - The name of the Chroma Cloud DB with the collection you want to browse. If not provided, the CLI will prompt you to select a DB from those available on your active [profile](./profile) . For local Chroma, the CLI uses the `default_database` .
- `local` - Instructs the CLI to find your collection on a local Chroma server at `http://localhost:8000` . If your local Chroma server is available on a different hostname, use the `host` argument instead.
- `host` - The host of your local Chroma server. This argument conflicts with `path` .
- `path` - The path of your local Chroma data. If provided, the CLI will use the data path to start a local Chroma server at an available port for browsing. This argument conflicts with `host` .
- `theme` - The theme of your terminal ( `light` or `dark` ). Optimizes the UI colors for your terminal’s theme. You only need to provide this argument once, and the CLI will persist it in `~/.chroma/config.json` .

cloud
cloud with DB
local default
local with host
local with path

```
chroma browse my-collection

```

### ​The Collection Browser UI

#### ​Main View

The main view of the Collection Browser shows you a tabular view of your data with record IDs, documents, and metadata. You can navigate the table using arrows, and expand each cell with
`Return`
. Only 100 records are loaded initially, and the next batch will load as you scroll down the table.

#### ​Search

You can enter the query editor by hitting
`s`
on the main view. This form allows you to submit
`.get()`
queries on your collection. You can edit the form by hitting
`e`
to enter edit mode, use
`space`
to toggle the metadata operator, and
`Esc`
to quit editing mode. To submit a query use
`Return`
.
The query editor persists your edits after you submit. You can clear it by hitting
`c`
. When viewing the results you can hit
`s`
to get back to the query editor, or
`Esc`
to get back to the main view.

[Suggest edits](https://github.com/chroma-core/chroma/edit/main/docs/mintlify/docs/cli/browse.mdx)
[Installing the CLIPrevious](/docs/cli/install)
[Copy CollectionsNext](/docs/cli/copy)
⌘
[Contact support](mailto:support@trychroma.com)