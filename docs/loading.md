# Loading Data
`fptools` offers robust data loading functionality for a variety of file types, including metadata injection, parallelism, caching, and preprocessing. Built-in are loaders for TDT tanks and Med-Associates files. You may also specify your own data loading functionality for arbitrary data which plugs into the `fptools` data loading infrastructure.

## Manifest
Provide a tabular data file (ex: xlsx, csv, tsv), and the fields in that row will be added to a given sessions metadata. When using the `load_data()` function, specify the path to your tabular data file using the `manifest_path` keyword argument. To correctly match a row from your manifest file to the correct `Session`, specify the column that will match the session name using the `manifest_index` keyword argument. For TDT sessions, this will be the block name, while for Med-associates sessions, this will be the file basename with no extensions. 

## Parallelism and Caching
Data loading can occur in parallel. Just specify the number of workers to use via the `max_workers` parameter. Each worker runs in a separate process, suitable for running preprocessing routines. The optimal number of workers would depend on the resources of the computer running the analysis.

Preprocessed data can be cached for quick retrieval later, without needing to re-perform expensive operations. To enable caching, set the `cache` parameter to `True`, setting to `False` will disable the cache. Cached data needs to be stored someplace on disk, and can be controlled by providing a filesystem path to the `cache_dir` parameter to a directory to contain the cache.

## Preprocessors
We offer several preprocessing routines you may choose from, or you may provide your own implementation; simply pass a something that implements the `Processor` protocol to the `preprocess` keyword argument. 

## DataLocators, Loaders and DataTypeAdaptors
The `load_data()` function takes a parameter `locator` which allows flexibility for finding a loading arbitrary data. For most users, the `locator` parameter can be set to the special strings `tdt`, `ma` or `auto` to find TDT blocks, med-associates data files, or a combination of the two, respectively.

For more advanced use cases, one may supply a function that implements the `DataLocator` protocol. The purpose of a `DataLocator` is to locate data, returning a list of `DataTypeAdaptor`s, with each `DataTypeAdaptor` corresponding to one `Session`. The `DataTypeAdaptor` should be populated with a `name` for the eventually created `Session`, a `path` to the data, and finally a list of one or more functions implementing the `Loader` protocol. A `Loader` receives a `Session` and path (from the `DataTypeAdaptor`) and is responsible for reading data from that path and populating the `Session` with the loaded data.

## Example
See the notebook `01_Data_Loading.ipynb` for an example of data loading.
