# DB Management - Chroma Docs

> Source: https://docs.trychroma.com/docs/cli/db

[Chroma Docshome page](/)
⌘
CLI
DB Management
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

The Chroma CLI lets you interact with your Chroma Cloud databases for your active
[profile](./profile)
.

### ​Connect

The
`connect`
command will output a connection code snippet for your Chroma Cloud database in Python or JS/TS. If you don’t provide the
`name`
or
`language`
the CLI will prompt you to choose your preferences. The
`name`
argument is always assumed to be the first, so you don’t need to include the
`--name`
flag.
The output code snippet will already have the API key of your profile set for the client construction.

```
chroma db connect [db_name] [--language python/JS/TS]

```

The
`connect`
command can also add Chroma environment variables (
`CHROMA_API_KEY`
,
`CHROMA_TENANT`
, and
`CHROMA_DATABASE`
) to a
`.env`
file in your current working directory. It will create a
`.env`
file for you if it doesn’t exist:

```
chroma db connect [db_name] --env-file

```

If you prefer to simply output these variables to your terminal use:

```
chroma db connect [db_name] --env-vars

```

Setting these environment variables will allow you to concisely instantiate the
`CloudClient`
with no arguments.

### ​Create

The
`create`
command lets you create a database on Chroma Cloud. It has the
`name`
argument, which is the name of the DB you want to create. If you don’t provide it, the CLI will prompt you to choose a name.
If a DB with your provided name already exists, the CLI will error.

```
chroma db create my-new-db

```

### ​Delete

The
`delete`
command deletes a Chroma Cloud DB. Use this command with caution as deleting a DB cannot be undone. The CLI will ask you to confirm that you want to delete the DB with the
`name`
you provided.

```
chroma db delete my-db

```

### ​List

The
`list`
command lists all the DBs you have under your current profile.

```
chroma db list

```

[Suggest edits](https://github.com/chroma-core/chroma/edit/main/docs/mintlify/docs/cli/db.mdx)
[Copy CollectionsPrevious](/docs/cli/copy)
[Sample AppsNext](/docs/cli/sample-apps)
⌘
[Contact support](mailto:support@trychroma.com)