# Python wrapper functions for the Spatial transcriptomics pipeline

from typing import Literal, Union, Optional
import scanpy as sc
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import squidpy as sq
import decoupler as dc
from pathlib import Path
from anndata import AnnData
from enum import Enum, unique
import chrysalis as ch


@unique
class Protocol(Enum):
    FF = "FF"
    FFPE = "FFPE"
    
@unique
class Organism(Enum):
    mouse = "mouse"
    human = "human"
    rat = "rat"
    

class Analyze():
    def __init__(self, protocol: Protocol, organism: Organism):
        self.protocol = protocol
        self.organism = organism

        if self.organism.value == Organism.human.value:
            self.mito_prefix = "MT-"
        elif self.organism.value == Organism.mouse.value:
            self.mito_prefix = "mt-"
        elif self.organism.value == Organism.rat.value:
            self.mito_prefix = "Mt-"
            
        else:
            self.mito_prefix = ""
        
    def __str__(self):
        return(f"Protocol: {self.protocol}, Organism: {self.organism}, Mitochondrial Gene Prefix: {self.mito_prefix}")

            
##########################
# General functions to import data
##########################


def generate_adata_objects(
    path: Union[str, Path],
    samples_names: list[str],
    metadata: pd.DataFrame(),
    analyze_params: Analyze,
    count_file: str = "filtered_feature_bc_matrix.h5",
    load_images: Optional[bool] = True,
) -> list[AnnData]:

    adata_objects = []

    for current_sample_name in samples_names:

        current_adata = sq.read.visium(
            path=os.path.join(path, current_sample_name),
            counts_file=count_file,
            load_images=load_images
            )
        current_adata.var_names_make_unique()
        # current_adata.obs["Sample_ID"] = current_sample_name
        current_metadata = metadata[metadata['readout_id'] == current_sample_name]

        if count_file == "filtered_feature_bc_matrix.h5": 
            current_adata.obs = current_adata.obs.merge(current_metadata, how='left', left_index=True, right_index=True)
            # It can be that there are some spots in the filtered matrix that we do not load into MongoDB because they have no expression. 
            # We have to remove these spots from the anndata object.
            current_adata = current_adata[current_adata.obs['readout_id'].notna(),:]
            
            
        # It may be that in the new probe sets versions for the FFPE protocol there are some 
        # mito genes, as we have seen in singlecell Flex experiements. I would therefore
        # remove the "if" for now for simplicity. To monitor potential differences between FF and
        # FFPE that may be worthy to include. 
        # if analyze_params.protocol.value == Protocol.FF.value:
        #    print("Fresh frozen samples: Detecting mitochondrial genes")
        # We also need to modify this to adapt to cyno where the mito names do not start with a prefix, but rather with a list of genes.
        current_adata.var["mt"] = current_adata.var_names.str.startswith(analyze_params.mito_prefix)
        sc.pp.calculate_qc_metrics(current_adata, qc_vars=["mt"], inplace=True)

        adata_objects.append(current_adata)

    return adata_objects


##########################
# QC related functions
##########################


def get_global_QCmetrics(
    path: str, samples_names: list[str], metrics_names: str = "metrics_summary.csv"
) -> pd.DataFrame():

    global_metrics = pd.DataFrame()
    for current_sample in samples_names:
        current_metrics = pd.read_csv(
            os.path.join(path, current_sample, metrics_names)
        )
        global_metrics = pd.concat([global_metrics, current_metrics])

    return global_metrics.set_index("Sample ID")


# We can simplify depending if the batch number comes in the name in a standard format
def get_barplot_qc(
    qc_df: pd.DataFrame(),
    color_by: list[str],
    variables_to_plot: np.ndarray,
    plots_row: Optional[int] = 3,
):

    for current_variable in variables_to_plot:

        current_df = qc_df[[current_variable]]
        current_df = current_df.assign(color_by=color_by)

        sns.set_theme(style="darkgrid")
        sns.barplot(
            data=current_df,
            x=current_variable,
            y=current_df.index,
            hue=current_df["color_by"],
            dodge=False,
        )
        # g.set_xticklabels(g.get_xticklabels(), rotation=30)
        plt.show()
        plt.close()


        
## Make the condition and the batch name not mandatory.         
def perform_qc_analysis(
    list_adata_filter: list[AnnData],
    list_adata_raw: list[AnnData],
    color_map="OrRd",
    sample_id="readout_id",
    condition_name="CONDITION",
    batch_name="Batch"
    
):

    for a in range(len(list_adata_filter)):

        sample = list_adata_filter[a].obs[sample_id].unique().tolist()
        condition = list_adata_filter[a].obs[condition_name].unique().tolist()
        batch =  list_adata_filter[a].obs[batch_name].unique().tolist()
        title = f"{sample_id}: {sample}, {condition_name}: {condition}, {batch_name}: {batch}"
        sc.pl.spatial(adata=list_adata_raw[a], img_key="hires", title=title, show=False)

        print(f"All spots: {title}")
        sc.pl.spatial(
            list_adata_raw[a],
            img_key="hires",
            color=["total_counts", "n_genes_by_counts"],
            color_map=color_map,
        )

        print(f"Tissue covered spots: {title}")
        sc.pl.spatial(
            list_adata_filter[a],
            img_key="hires",
            color=["total_counts", "n_genes_by_counts"],
            color_map=color_map,
        )

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        sns.violinplot(
            data=list_adata_raw[a].obs,
            x="in_tissue",
            y="total_counts",
            inner="quart",
            linewidth=1,
            ax=axs[0],
        )
        sns.stripplot(
            y="total_counts",
            x="in_tissue",
            data=list_adata_raw[a].obs,
            color="black",
            edgecolor="gray",
            size=2.5,
            ax=axs[0],
        )

        sns.violinplot(
            data=list_adata_raw[a].obs,
            x="in_tissue",
            y="n_genes_by_counts",
            inner="quart",
            linewidth=1,
            ax=axs[1],
        )
        sns.stripplot(
            y="n_genes_by_counts",
            x="in_tissue",
            data=list_adata_raw[a].obs,
            color="black",
            edgecolor="gray",
            size=2.5,
            ax=axs[1],
        )
        plt.show(fig)
        plt.close(fig)

        fig, axs = plt.subplots(1, 3, figsize=(22, 5))
        sns.scatterplot(
            data=list_adata_filter[a].obs,
            x="total_counts",
            y="n_genes_by_counts",
            ax=axs[0],
        )
        sns.violinplot(
            data=list_adata_filter[a].obs,
            x=sample_id,
            y="total_counts",
            inner="quart",
            linewidth=1,
            ax=axs[1],
        )
        sns.stripplot(
            y="total_counts",
            x=sample_id,
            data=list_adata_filter[a].obs,
            color="black",
            edgecolor="gray",
            size=2.5,
            ax=axs[1],
        )

        sns.violinplot(
            data=list_adata_filter[a].obs,
            x=sample_id,
            y="n_genes_by_counts",
            inner="quart",
            linewidth=1,
            ax=axs[2],
        )
        sns.stripplot(
            y="n_genes_by_counts",
            x=sample_id,
            data=list_adata_filter[a].obs,
            color="black",
            edgecolor="gray",
            size=2.5,
            ax=axs[2],
        )
        plt.show(fig)
        plt.close(fig)

        # TODO: Check for the FFPE Implementation
        print(f"Mithocondrial genes: {title}")
        sc.pl.spatial(
            list_adata_filter[a],
            img_key="hires",
            color=["total_counts_mt", "pct_counts_mt"],
            color_map=color_map,
        )

        fig, axs = plt.subplots(1, 3, figsize=(22, 5))
        sns.scatterplot(
            data=list_adata_filter[a].obs,
            x="total_counts",
            y="total_counts_mt",
            ax=axs[0],
        )

        sns.violinplot(
            data=list_adata_filter[a].obs,
            x=sample_id,
            y="total_counts_mt",
            inner="quart",
            linewidth=1,
            ax=axs[1],
        )
        sns.stripplot(
            y="total_counts_mt",
            x=sample_id,
            data=list_adata_filter[a].obs,
            color="black",
            edgecolor="gray",
            size=2.5,
            ax=axs[1],
        )

        sns.violinplot(
            data=list_adata_filter[a].obs,
            x=sample_id,
            y="pct_counts_mt",
            inner="quart",
            linewidth=1,
            ax=axs[2],
        )
        sns.stripplot(
            y="pct_counts_mt",
            x=sample_id,
            data=list_adata_filter[a].obs,
            color="black",
            edgecolor="gray",
            size=2.5,
            ax=axs[2],
        )
        plt.show(fig)
        plt.close(fig)


# By default, I am entering very "relaxed" thresholds
def qc_filtering(
    list_adatas: list[AnnData],
    min_counts: Optional[int] = 1000,
    max_counts: Optional[int] = 40000,
    mt_pct_content: Optional[int] = 20,
    min_cells: Optional[int] = 5,
    sample_ID: Optional[str] = 'readout_id',
) -> list[AnnData]:

    adata_filtered_objects: list[AnnData] = []

    for current_adata in list_adatas:

        current_sample = np.asarray(current_adata.obs[sample_ID].unique())
        print(current_sample)

        print(f"# Spots before filter: {current_adata.n_obs}")
        print(f"# Genes before filter: {current_adata.n_vars}")

        sc.pp.filter_cells(current_adata, min_counts=min_counts, inplace=True)
        sc.pp.filter_cells(current_adata, max_counts=max_counts, inplace=True)
        current_adata = current_adata[
            current_adata.obs["pct_counts_mt"] < mt_pct_content
        ]
        sc.pp.filter_genes(current_adata, min_cells=min_cells, inplace=True)

        print(f"# Spots after filter: {current_adata.n_obs}")
        print(f"# Genes before filter: {current_adata.n_vars}")

        adata_filtered_objects.append(current_adata)

    return adata_filtered_objects


##########################
# Normalization, manifold embedding, clustering and Marker genes
##########################

# TODO: more parameters to be added.
def norm_hvg(
    list_adatas: list[AnnData],
    flavor: Literal["seurat", "cell_ranger", "seurat_v3"] = "seurat",
    n_top_genes: Optional[int] = None,
) -> list[AnnData]:

    adata_objects: list[AnnData] = []

    for adata in list_adatas:
        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, flavor=flavor, n_top_genes=n_top_genes)
        adata_objects.append(adata)

    return adata_objects


def cluster_umap(
    list_adatas: list[AnnData],
    n_comps: Optional[int] = None,
    use_highly_variable: Optional[bool] = None,
    n_neighbors: Optional[int] = 15,
    n_pcs: Optional[int] = None,
    min_dist: Optional[float] = 0.5,
    spread: Optional[float] = 1.0,
    n_components: Optional[int] = 2,
    key_added: str = "clusters",
    resolution: float = 0.75,
    sample_id: Optional[str] = "readout_id",
) -> list[AnnData]:

    adata_objects: list[AnnData] = []

    for current_adata in list_adatas:

        current_sample = np.asarray(current_adata.obs[sample_id].unique())
        print(current_sample)

        sc.pp.pca(
            current_adata, n_comps=n_comps, use_highly_variable=use_highly_variable
        )
        sc.pp.neighbors(current_adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
        sc.tl.umap(
            current_adata, min_dist=min_dist, spread=spread, n_components=n_components
        )
        sc.tl.leiden(current_adata, resolution=resolution, key_added=key_added)

        plt.rcParams["figure.figsize"] = (4, 4)
        sc.pl.umap(
            current_adata,
            color=["total_counts", "n_genes_by_counts", key_added],
            wspace=0.4,
        )

        plt.rcParams["figure.figsize"] = (6, 6)
        sc.pl.spatial(current_adata, img_key="hires", color=key_added, size=1.5)

        adata_objects.append(current_adata)

    return adata_objects


def get_markers_clusters(
    list_adatas,
    group_by="clusters",
    groups_1="all",
    reference="rest",
    method="wilcoxon",
    corr_method="benjamini-hochberg",
    groups_2=None,
    n_genes=5,
    genes_visualize=20,
    sample_id: Optional[str] = "readout_id",
) -> list[AnnData]:

    adata_objects: list[AnnData] = []

    for current_adata in list_adatas:

        current_sample = np.asarray(current_adata.obs[sample_id].unique())
        print(current_sample)

        sc.tl.rank_genes_groups(
            current_adata,
            group_by,
            groups=groups_1,
            reference=reference,
            method=method,
            corr_method=corr_method,
        )
        plt.rcParams["figure.figsize"] = (6, 6)
        sc.pl.rank_genes_groups_heatmap(
            current_adata, groupby=group_by, groups=groups_2, n_genes=n_genes
        )

        # plt.rcParams["figure.figsize"] = (6, 6)
        # sc.pl.spatial(current_adata, img_key="hires", color=["clusters"], size = 1.5)

        group = list(current_adata.uns["rank_genes_groups"]["names"].dtype.names)
        colnames = ["names", "scores", "logfoldchanges", "pvals", "pvals_adj"]
        d = [
            pd.DataFrame(current_adata.uns["rank_genes_groups"][c])[group]
            for c in colnames
        ]
        d = pd.concat(d, axis=1, names=[None, "group"], keys=colnames)
        d = d.stack(level=1).reset_index()
        d["group"] = pd.Categorical(d["group"], categories=group)
        d = d.drop(columns="level_0")
        d = d.sort_values(["pvals_adj"], ascending=True)

        print(d.head(genes_visualize))

        adata_objects.append(current_adata)

    return adata_objects


##########################
# Spatially variable genes
##########################


# max_neighs Only used for Septal
def get_sp_variable_genes(
    list_adatas,
    method="moran",
    min_number_spots=100,
    number_hvg=100,
    n_perms=100,
    n_jobs=1,
    genes_visualize=10,
    max_neighs=6,
) -> list[AnnData]:

    adata_objects: list[AnnData] = []

    for current_adata in list_adatas:

        current_sample = np.asarray(current_adata.obs["Sample_ID"].unique())
        print(current_sample)

        genes = current_adata.var_names[
            (current_adata.var.n_cells > min_number_spots)
            & current_adata.var.highly_variable
        ][0:number_hvg]
        sq.gr.spatial_neighbors(current_adata)
        genes = current_adata.var_names[
            (current_adata.var.n_cells > min_number_spots)
            & current_adata.var.highly_variable
        ][0:number_hvg]

        if method == "sepal":
            sq.gr.sepal(
                current_adata, max_neighs=max_neighs, genes=genes, n_jobs=n_jobs
            )
            print(current_adata.uns["sepal_score"].head(genes_visualize))
        elif method == "moran":
            sq.gr.spatial_autocorr(
                current_adata, mode="moran", genes=genes, n_perms=n_perms, n_jobs=n_jobs
            )
            print(current_sample)
            print(current_adata.uns["moranI"].head(genes_visualize))
        else:
            print("Unknown method")

        adata_objects.append(current_adata)

    return adata_objects


##########################
# Footprint-based methods
##########################


def get_pathway_activity(
    list_adatas,
    organism="human",
    top_genes=500,
    verbose=False,
    groupby="clusters",
    vmin=-2,
    vmax=2,
    cmap="coolwarm",
    use_raw=False,
) -> list[AnnData]:

    adata_objects: list[AnnData] = []
    model = dc.get_progeny(organism=organism, top=top_genes)

    for a in range(len(list_adatas)):

        current_sample = np.asarray(list_adatas[a].obs["Sample_ID"].unique())
        print(current_sample)

        current_adata = list_adatas[a]

        dc.run_mlm(
            mat=current_adata,
            net=model,
            source="source",
            target="target",
            weight="weight",
            verbose=verbose,
            use_raw=use_raw,
        )

        current_adata.obsm["progeny_mlm_estimate"] = current_adata.obsm[
            "mlm_estimate"
        ].copy()
        current_adata.obsm["progeny_mlm_pvals"] = current_adata.obsm["mlm_pvals"].copy()

        acts = dc.get_acts(current_adata, obsm_key="progeny_mlm_estimate")

        mean_acts = dc.summarize_acts(acts, groupby=groupby, min_std=0)
        print(mean_acts)

        sns.clustermap(
            mean_acts, xticklabels=mean_acts.columns, vmin=vmin, vmax=vmax, cmap=cmap
        )
        plt.show()

        adata_objects.append(current_adata)

    return adata_objects


def get_TF_activity(
    list_adatas,
    organism="human",
    levels=["A", "B", "C"],
    min_n=5,
    min_std=0.75,
    verbose=False,
    groupby="clusters",
    vmin=-2,
    vmax=2,
    cmap="coolwarm",
    use_raw=False,
) -> list[AnnData]:

    adata_objects: list[AnnData] = []
    net = dc.get_dorothea(organism=organism, levels=["A", "B", "C"])

    for a in range(len(list_adatas)):

        current_sample = np.asarray(list_adatas[a].obs["Sample_ID"].unique())
        print(current_sample)

        current_adata = list_adatas[a]

        dc.run_mlm(
            mat=current_adata,
            net=net,
            min_n=min_n,
            source="source",
            target="target",
            weight="weight",
            verbose=verbose,
            use_raw=use_raw,
        )

        current_adata.obsm["dorothea_mlm_estimate"] = current_adata.obsm[
            "mlm_estimate"
        ].copy()
        current_adata.obsm["dorothea_mlm_pvals"] = current_adata.obsm[
            "mlm_pvals"
        ].copy()

        acts = dc.get_acts(current_adata, obsm_key="dorothea_mlm_estimate")

        mean_acts = dc.summarize_acts(acts, groupby=groupby, min_std=min_std)
        print(mean_acts)

        sns.clustermap(
            mean_acts, xticklabels=mean_acts.columns, vmin=vmin, vmax=vmax, cmap=cmap
        )
        plt.show()

        adata_objects.append(current_adata)

    return adata_objects


#### Copied from Besca https://github.com/bedapub/besca/blob/master/besca/export/_export.py

## overwriding _field_template to avoid scientific notations
from scipy import io, sparse
from scipy.io.mmio import MMFile

def export_cp10k(adata, basepath, write_metadata= True, geneannotation="SYMBOL", additional_geneannotation="ENSEMBL"):
    """Export raw cp10k to FAIR format for loading into database

    wrapper function for X_to_mtx with correct folder structure for loading into database.

    parameters
    ----------
    adata: `AnnData`
        AnnData object that is to be exported
    basepath: `str`
        root path to the Analysis folder (i.e. ../analyzed/<ANALYSIS_NAME>)

    returns
    -------
    None
        writes to file

    """

    # call wrapper function
    X_to_mtx(
        adata=adata,
        outpath=os.path.join(basepath, "normalized_counts", "cp10k"),
        write_metadata=write_metadata,
        geneannotation=geneannotation,
        additional_geneannotation=additional_geneannotation,
    )    

class MMFileFixedFormat(MMFile):
    def _field_template(self, field, precision):
        # Override MMFile._field_template.
        return f"%.{precision}f\n"

def X_to_mtx(
    adata: AnnData,
    outpath: str = None,
    write_metadata: bool = True,
    geneannotation: str = "SYMBOL",
    additional_geneannotation: str = "ENSEMBL",
) -> None:
    """export adata object to mtx format (matrix.mtx, genes.tsv, barcodes.tsv)

    exports the counts contained in adata.X to a matrix.mtx file (in sparse format),
    genes taken from adata.var_names (and if applicable adata.var) to genes.tsv and the
    cellbarcodes from adata.obs_names to barcodes.tsv. If annotation = True, then the entire
    pd.Dataframe contained in adata.obs is in addition exported to metadata.tsv.
    Through the parameter geneannotation you can specify which type of geneidentifer is saved in
    adata.var_names. You can pass an additional string to the parameter additional_geneannotation
    which specifies under which column name in adata.var an additional geneannotation can be
    found. Currently this function is only capable of dealing with geneannotation of the type
    ENSEMBL or SYMBOL. This feature is intended to conserve the correct order of ENSEMBL IDs and
    SYMBOLS in the genes.tsv file.
    If the outpath directory does not exist, this function automatically generates it.

    parameters
    ----------
    adata: AnnData
        the AnnData object that should be exported
    outpath `str` | default = current working directory
        filepath to the directory in which the results should be outputed, if no directory is
        specified it outputs the results to the current working directory.
    write_metadata: `bool` | default = False
        boolian indicator if the annotation contained in adata.obs should be exported as well
    geneannotation: `'ENSEMBL'` or `'SYMBOL'`
        string indicator of the type of gene annotation saved in adata.var_names
    additional_geneannotation: `str` | default = None
        string identifying the coloumn name in which either the SYMBOL or the ENSEMBL geneids
        are contained as additional gene annotation in adata.var

    returns
    -------
    None
        writes out files to the specified output directory

    """
    if outpath is None:
        outpath = os.getcwd()
    ### check if the outdir exists if not create
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    ### write out matrix.mtx as float with 3 significant digits
    print("writing out matrix.mtx ...")

    if type(adata.X) == np.ndarray:
        E = sparse.csr_matrix(adata.X).T
    else:
        E = adata.X.tocsr().T  # transpose back into the same format as we imported it

    MMFileFixedFormat().write(os.path.join(outpath, "matrix.mtx"), a=E, precision=2)
    print("adata.X successfully written to matrix.mtx.")
    ### export genes

    # get genes to export
    if geneannotation == "SYMBOL":
        genes_SYMBOL = adata.var_names.tolist()
        # get additional annotation saved in adata.var
        if additional_geneannotation is not None:
            genes_ENSEMBL = adata.var.get(additional_geneannotation)
        else:
            print(
                "No ENSEMBL gene ids provided, Besca will fill the respective columns in genes.tsv with NA"
            )
            genes_ENSEMBL = ["NA"] * len(genes_SYMBOL)

    elif geneannotation == "ENSEMBL":
        genes_ENSEMBL = adata.var_names.tolist()
        # get additional annotation saved in adata.var
        if additional_geneannotation is not None:
            genes_SYMBOL = adata.var.get(additional_geneannotation)
        else:
            print(
                "No SYMBOLS provided, Besca will fill the respective columns in genes.tsv with NA"
            )
            genes_SYMBOL = ["NA"] * len(genes_ENSEMBL)

    else:
        sys.exit("need to provide either 'ENSEMBL' or 'SYMBOL' gene annotation.")

    feature = None
    # add catch to write out type of annotation
    if "feature_type" in adata.var.columns:
        print("feature annotation is present and will be written out")
        feature = True
        gene_feature = adata.var.get("feature_type")

    # write the genes out in the correct format (first ENSEMBL THEN SYMBOL)
    with open(os.path.join(outpath, "genes.tsv"), "w") as fp:
        if feature is not None:
            for ENSEMBL, symbol, feature in zip(
                genes_ENSEMBL, genes_SYMBOL, gene_feature
            ):
                fp.write(ENSEMBL + "\t" + symbol + "\t" + feature + "\n")
        else:
            for ENSEMBL, symbol in zip(genes_ENSEMBL, genes_SYMBOL):
                fp.write(ENSEMBL + "\t" + symbol + "\n")
        fp.close()
        print("genes successfully written out to genes.tsv")

    ### write out the cellbarcodes
    # cellbarcodes = adata.obs_names.tolist()
    cellbarcodes = adata.obs['CELL'].tolist()
    with open(os.path.join(outpath, "barcodes.tsv"), "w") as fp:
        for barcode in cellbarcodes:
            fp.write(barcode + "\n")
        fp.close()
        print("cellbarcodes successfully written out to barcodes.tsv")

    ### export annotation
    if write_metadata == True:
        annotation = adata.obs
        annotation.to_csv(
            os.path.join(outpath, "metadata.tsv"), sep="\t", header=True, index=True
        )
        print("annotation successfully written out to metadata.tsv")

    return None
    sys.exit(0)
    

    
## Modified clustering (original function name) by export_clustering. 
def labeling_info(
    outpath: str = None,
    description: str = "leiden clustering",
    public: bool = False,
    default: bool = True,
    expert: bool = False,
    reference: bool = False,
    method: str = "leiden",
    annotated_version_of: str = "-",
    filename: str = "labelinfo.tsv",
) -> None:
    """write out labeling info for uploading to database

    This functions outputs the file labelinfo.tsv which is needed to annotate a written out
    labeling in the scseq database.

    parameters
    ----------
    outpath: `str` | default = current working directory
        The filepath as a string indicating the location where the file should be written out to.
    description: `str` | default = 'leiden clustering'
        string describing what type of information is saved in the corresponding labeling.
    public: `bool` | default = False
        boolian indicator if the contained labeling information is available in the public domain.
    default_ `bool` | default = True
        boolian indicator if the labeling was created using a standardized process e.g. the leiden
        clusters outputed by the standard pipeline (this should be false if expert is true)
    expert: `bool` | default = False
        boolian indicator if the labeling was created by an 'expert' i.e. manually done (this should
        be false if default is true)
    reference: `bool` | default = True
        boolian indicator if this is the labeling (e.g. celltype annotation) that should be used for further analysis
        (there should only be one reference labeling per study)
    method: `str` | default = 'leiden'
        string indicating the type of method that was applied, e.g. if the labeling is of a clustering
        which clustering algorithm was used.
    annotated_version_of: `str` | default = '-'
        string identifying of what othe labeling/dataset this is an annotated version of (so for
        example if the labeling is celltype annotation of a leiden clustering then this would
        reference the leiden clsutering that was used to obtain the clusters that were then
        labeled here)
    filename: `str` | default = 'labelinfo.tsv'
        string indicating the filename that should be used. This is per default set to the correct
        file name for uploading to the scseq database.

    returns
    -------
    None
        results are written out to a file instead

    """
    if outpath is None:
        outpath = os.getcwd()
    if public:
        Public = "TRUE"
    else:
        Public = "FALSE"

    if default:
        Default = "TRUE"
    else:
        Default = "FALSE"

    if expert:
        Expert = "TRUE"
    else:
        Expert = "FALSE"

    if reference:
        Reference = "TRUE"
    else:
        Reference = "FALSE"

    ciFile = os.path.join(outpath, filename)
    with open(ciFile, "w") as fp:
        fp.write(
            "description\tisPublic\tisDefault\tisExpert\tisReference\tmethod\tannotated_version_of\n"
        )
        fp.write(
            description
            + "\t"
            + Public
            + "\t"
            + Default
            + "\t"
            + Expert
            + "\t"
            + Reference
            + "\t"
            + method
            + "\t"
            + annotated_version_of
            + "\n"
        )
    fp.close()
    print(f"{filename} successfully written out")

    return None
    sys.exit(0)


def export_clustering(adata, basepath, method = 'leiden', use_raw = True):
    """Export cluster to cell mapping to FAIR format for loading into database

    wrapper function for louvain and labeling_info with correct folder structure/names
    for loading into the database.

    parameters
    ----------
    adata: `AnnData`
        AnnData object that is to be exported
    basepath: `str`
        root path to the Analysis folder (i.e. ../analyzed/<ANALYSIS_NAME>)
    method: `str`
        method of clustering used previously, should be leiden or louvain

    returns
    -------
    None
        writes to file
    """

    clustering(adata, outpath=os.path.join(basepath, "labelings", method), method=method, use_raw=use_raw)
    labeling_info(
        outpath=os.path.join(basepath, "labelings", method),
        description=method + " clustering",
        method=method,
    )


def clustering(
    adata: AnnData,
    outpath: str = None,
    export_average: bool = True,
    export_fractpos: bool = True,
    method: str = "leiden",
    use_raw: bool = True,
):
    """export mapping of cells to clusters to .tsv file

    This function exports the labels saved in adata.obs.method and the corresponding cell barcode to the file cell2labels.tsv.

    parameters
    ----------
    adata: `AnnData`
        the AnnData object containing the clusters
    outpath: `str` | default = current working directory
        filepath to the directory in which the results should be outputed, if no directory is
        specified it outputs the results to the current working directory.
    export_average: `bool` | default = True
        boolian indicator if the average gene expression of each cluster should be exported to file
    export_fractpos: `bool` | default = True
        boolian indicator if the fraction of positive cells (i.e. cells that express the gene) should
        be exported to file
    method: `str1 | default = 'leiden'
        string indicating the clustering method used and where to store the results. Shuold be either louvain or leiden
    returns
    -------
    None
        files are written out.

    """
    if outpath is None:
        outpath = os.getcwd()
    if not method in ["leiden", "louvain"]:
        raise ValueError("method argument should be leiden or louvain")
    cluster_data = adata.obs.get(method).to_frame(name="LABEL")
    if cluster_data is None:
        sys.exit("need to perform " + method + " clustering before exporting")
    # perform export calling the general export function
    write_labeling_to_files(
        adata,
        outpath=outpath,
        column=method,
        label="LABEL",
        filename="cell2labels.tsv",
        export_average=export_average,
        export_fractpos=export_fractpos,
        use_raw=use_raw,
    )
    return None    

def write_labeling_to_files(
    # Slightly modify
    adata_input: AnnData,
    outpath: str = None,
    column: str = "leiden",
    label: str = "LABEL",
    filename: str = "cell2labels.tsv",
    export_average: bool = True,
    export_fractpos: bool = True,
    use_raw: bool = True,
) -> None:
    """export mapping of cells to specified label to .tsv file

    This is a function with which any type of labeling (i.e. celltype annotation, leiden
    clustering, etc.) can be written out to a .tsv file. The generated file can then also be easily
    uploaded to the database since it fullfilles the FAIR document standards.

    To ensure FAIR compatbility label, and file name should not be changed.

    parameters
    ----------
    adata_input: `AnnData`
        the AnnData object containing the label
    outpath `str` | default = current working directory
        filepath to the directory in which the results should be outputed, if no directory is
        specified it outputs the results to the current working directory.
    column: `str` | default = 'leiden'
        Name of the column in adata.obs that is to be mapped to cell barcodes and written out to file.
    label: `str` | default = 'LABEL'
        label above the column when it is written out to file
    filename: `str` | default = 'cell2labels.tsv'
        Filename that is written out.

    returns
    -------
    None
        files are written out.

    """
    if outpath is None:
        outpath = os.getcwd()

    ## We change the index of the adata.obs object to have the sequencial Cell ID instead of the sample and barcode name.
    
    adata = adata_input.copy()
    adata.obs = adata.obs.reset_index().set_index('CELL')
    
    data = adata.obs.get(column)
    if data is None:
        sys.exit("please specify a column name that is present in adata.obs")

    data = adata.obs.get(column).to_frame(name=label)

    ### check if the outdir exists if not create
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    data.to_csv(
        os.path.join(outpath, filename), sep="\t", header=True, index_label="CELL"
    )
    print("mapping of cells to ", column, "exported successfully to", filename)

    if export_average:
        if column == "louvain" or column == "leiden":
            labeling = sorted(set(adata.obs[column].astype(int)))
        else:
            labeling = sorted(set(adata.obs[column]))

        label_names = []
        for i in range(len(labeling)):
            label_names.append(str(labeling[i]))

        num_labels = len(label_names)

        if use_raw:
            gct = pd.DataFrame(index=adata.raw.var_names, columns=label_names)
            mydat = adata.raw.var.copy()
            if type(adata.raw.X) == np.ndarray:
                E = sparse.csr_matrix(adata.raw.X).T
            else:
                E = adata.raw.X.tocsr().T

        else:
            gct = pd.DataFrame(index=adata.var_names, columns=label_names)
            mydat = adata.var.copy()
            if type(adata.X) == np.ndarray:
                E = sparse.csr_matrix(adata.X).T
            else:
                E = adata.X.tocsr().T

        # revert to linear scale
        E = E.expm1()

        for i in range(num_labels):
            cells = np.where(adata.obs.get(column) == label_names[i])[0]
            gct.iloc[:, i] = E[:, cells].mean(axis=1)  # get mean expression per gene
            labeling_size = len(cells)

        mydat = mydat.loc[:, ["SYMBOL", "ENSEMBL"]]
        mydat.rename(columns={"SYMBOL": "Description"}, inplace=True)
        gct = mydat.merge(gct, how="right", left_index=True, right_index=True)

        gct.set_index("ENSEMBL", inplace=True)
        gct.index.names = ["NAME"]

        # write out average expression
        gctFile_average = os.path.join(outpath, "average.gct")
        with open(gctFile_average, "w") as fp:
            fp.write("#1.2" + "\n")
            fp.write(
                str(gct.shape[0]) + "\t" + str(gct.shape[1] - 1) + "\n"
            )  # "description" already merged in as a column
        fp.close()
        # ...and then the matrix
        gct.to_csv(
            gctFile_average,
            sep="\t",
            index=True,
            index_label="NAME",
            header=True,
            mode="a",
            float_format="%.3f",
        )
        print("average.gct exported successfully to file")

    if export_fractpos:
        if column == "louvain" or column == "leiden":
            labeling = sorted(set(adata.obs[column].astype(int)))
        else:
            labeling = sorted(set(adata.obs[column]))

        label_names = []
        for i in range(len(labeling)):
            label_names.append(str(labeling[i]))

        num_labels = len(label_names)

        if use_raw:
            f = pd.DataFrame(index=adata.raw.var_names, columns=label_names)
            mydat = adata.raw.var.copy()
            if type(adata.raw.X) == np.ndarray:
                E = sparse.csr_matrix(adata.raw.X).T
            else:
                E = adata.raw.X.tocsr().T
        else:
            gct = pd.DataFrame(index=adata.var_names, columns=label_names)
            f = pd.DataFrame(index=adata.var_names, columns=label_names)
            mydat = adata.var.copy()
            if type(adata.X) == np.ndarray:
                E = sparse.csr_matrix(adata.X).T
            else:
                E = adata.X.tocsr().T
        # revert to linear scale
        E = E.expm1()

        for i in range(num_labels):
            cells = np.where(adata.obs.get(column) == label_names[i])[0]
            a = E[:, cells].getnnz(axis=1)  # get number of values not 0
            f.iloc[:, i] = a.copy()
        f = f.astype(float)
        for i in range(num_labels):
            cells = np.where(adata.obs.get(column) == label_names[i])[0]
            labeling_size = len(cells)
            f[label_names[i]] = f[label_names[i]] / labeling_size

        mydat = mydat.loc[:, ["SYMBOL", "ENSEMBL"]]
        mydat.rename(columns={"SYMBOL": "Description"}, inplace=True)

        f = mydat.merge(f, how="right", left_index=True, right_index=True)
        f.set_index("ENSEMBL", inplace=True)
        f.index.names = ["NAME"]

        # write out frac_pos.gct
        gctFile_fracpos = os.path.join(outpath, "fract_pos.gct")
        with open(gctFile_fracpos, "w") as fp:
            fp.write("#1.2" + "\n")
            fp.write(
                str(f.shape[0]) + "\t" + str(f.shape[1] - 1) + "\n"
            )  # "description" already merged in as a column
        fp.close()
        # ...and then the matrix
        f.to_csv(
            gctFile_fracpos,
            sep="\t",
            index=True,
            index_label="NAME",
            header=True,
            mode="a",
            float_format="%.3f",
        )
        print("fract_pos.gct exported successfully to file")

    return None
    sys.exit(0)
    
    
def export_rank(adata, basepath, type="wilcox", labeling_name="leiden", geneannotation: str = "SYMBOL", additional_geneannotation: str = "ENSEMBL"):
    """Export ranked genes to FAIR format for loading into database

    wrapper function for ranked_genes with correct folder structure/names
    for loading into the database.

    parameters
    ----------
    adata: `AnnData`
        AnnData object that is to be exported
    basepath: `str`
        root path to the Analysis folder (i.e. ../analyzed/<ANALYSIS_NAME>)
    type: `str` | default = 'wilcox'
        indicator of the statistical method employed, can be one of: 'wilcox' or 't-test overest var'  or 't-test'
    labelingname: `str` | default = `louvain`
        labeling that will be exported

    returns
    -------
    None
        writes to file

    """
    # export rank files
    ranked_genes(
        adata=adata, outpath=os.path.join(basepath, "labelings", labeling_name), geneannotation = geneannotation, additional_geneannotation = additional_geneannotation, type=type
    )
    
def ranked_genes(
    adata: AnnData,
    type: str = "wilcox",
    outpath: str = None,
    geneannotation: str = "SYMBOL",
    additional_geneannotation: str = "ENSEMBL",
):
    """export marker genes for each cluster to .gct file

    This function exports the results of scanpy.tl.rank_genes_groups() on your AnnData object to a .gct
    file. This file can easily be uploaded into the scsqe database since it follows the FAIR data
    formats. It expect the label "rank_genes_groups" and not a personalized one.

    A prerequisit for executing this function is that sc.tl.rank_genes_groups() has already been run.
    Through the variables geneannotation and additional_geneannotation you can specify the type of
    gene annotationi that is saved in adata.var_names and any additional geneannotation columns saved
    in adata.vars.

    parameters
    ----------
    adata: `AnnData`
        AnnData object on which scanpy.tl.rank_genes_groups has been executed
    type: `str` | 'wilcox' or 't-test overest var'  or 't-test'
    outpath `str` | default = current working directory
        filepath to the directory in which the results should be outputed, if no directory is
        specified it outputs the results to the current working directory.
    geneannotation: `'ENSEMBL'` or `'SYMBOL'`
        type of gene annotation that is located in adata.var_names
    additional_geneannotation:

    returns
    -------
    None
        writes results to file in output directory

    """
    if outpath is None:
        outpath = os.getcwd()
    if adata.uns.get("rank_genes_groups") is None:
        sys.exit(
            "need to rank genes before export, please run: scanpy.tl.rank_genes() before proceeding with export"
        )
    else:
        # extract relevant data from adata object
        rank_genes = adata.uns["rank_genes_groups"]

    # get group names
    groups = rank_genes["names"].dtype.names

    # get number of groups
    group_number = len(groups)

    # get gene information
    mydat = adata.raw.var.loc[:, ["SYMBOL", "ENSEMBL"]]
    mydat.rename(columns={"SYMBOL": "Description"}, inplace=True)

    # initialize dataframe for storing the scores
    scores = pd.DataFrame(
        {"n_" + key[:1]: rank_genes[key][groups[0]] for key in ["names", "scores"]}
    )
    scores.rename(columns={"n_s": groups[0], "n_n": "NAME"}, inplace=True)
    scores.set_index("NAME", inplace=True)
    scores.index = scores.index.astype(str)

    for i in range(1, group_number):
        n_scores = pd.DataFrame(
            {"n_" + key[:1]: rank_genes[key][groups[i]] for key in ["names", "scores"]}
        )
        n_scores.set_index("n_n", inplace=True)
        n_scores.index = n_scores.index.astype(str)
        scores = scores.merge(n_scores, how="left", left_index=True, right_index=True)
        newname = groups[i]
        scores.rename(columns={"n_s": newname}, inplace=True)
    scores = scores.astype(float)

    # merge in gene annotation
    scores = mydat.merge(scores, how="right", left_index=True, right_index=True)

    # make index into ENSEMBL instead of symbol
    scores.set_index("ENSEMBL", inplace=True)
    scores.index.names = ["NAME"]

    # get pvalues
    # initialize dataframe for storing the pvalues
    pvalues = pd.DataFrame(
        {"n_" + key[:1]: rank_genes[key][groups[0]] for key in ["names", "pvals"]}
    )
    pvalues.rename(columns={"n_p": groups[0], "n_n": "NAME"}, inplace=True)
    pvalues.set_index("NAME", inplace=True)
    pvalues.index = pvalues.index.astype(str)

    for i in range(1, group_number):
        n_pvalues = pd.DataFrame(
            {"n_" + key[:1]: rank_genes[key][groups[i]] for key in ["names", "pvals"]}
        )
        n_pvalues.set_index("n_n", inplace=True)
        n_pvalues.index = n_pvalues.index.astype(str)
        pvalues = pvalues.merge(
            n_pvalues, how="left", left_index=True, right_index=True
        )
        newname = groups[i]
        pvalues.rename(columns={"n_p": newname}, inplace=True)
    pvalues = pvalues.astype(float)

    # merge in pvalues
    pvalues = mydat.merge(pvalues, how="right", left_index=True, right_index=True)

    # make index into ENSEMBL instead of symbol
    pvalues.set_index("ENSEMBL", inplace=True)
    pvalues.index.names = ["NAME"]

    logFC = pd.DataFrame(
        {
            "n_" + key[:1]: rank_genes[key][groups[0]]
            for key in ["names", "logfoldchanges"]
        }
    )
    logFC.rename(columns={"n_l": groups[0], "n_n": "NAME"}, inplace=True)
    logFC.set_index("NAME", inplace=True)
    logFC.index = logFC.index.astype(str)

    for i in range(1, group_number):
        n_logFC = pd.DataFrame(
            {
                "n_" + key[:1]: rank_genes[key][groups[i]]
                for key in ["names", "logfoldchanges"]
            }
        )
        n_logFC.set_index("n_n", inplace=True)
        n_logFC.index = n_logFC.index.astype(str)
        logFC = logFC.merge(n_logFC, how="left", left_index=True, right_index=True)
        newname = groups[i]
        logFC.rename(columns={"n_l": newname}, inplace=True)
    logFC = logFC.astype(float)

    # merge in pvalues
    logFC = mydat.merge(logFC, how="right", left_index=True, right_index=True)

    # make index into ENSEMBL instead of symbol
    logFC.set_index("ENSEMBL", inplace=True)
    logFC.index.names = ["NAME"]

    ### check if the outdir exists if not create
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    if type == "wilcox":
        gct_rank_File = os.path.join(outpath, "WilxRank.gct")
        gct_pvalue_File = os.path.join(outpath, "WilxRank.pvalues.gct")
        gct_logFC_File = os.path.join(outpath, "WilxRank.logFC.gct")
    elif type == "t-test overest var":
        gct_rank_File = os.path.join(outpath, "tTestRank.gct")
        gct_pvalue_File = os.path.join(outpath, "tTestRank.pvalues.gct")
        gct_logFC_File = os.path.join(outpath, "tTestRank.logFC.gct")
    elif type == "t-test":
        gct_rank_File = os.path.join(outpath, "OverestVarRank.gct")
        gct_pvalue_File = os.path.join(outpath, "OverestVar.pvalues.gct")
        gct_logFC_File = os.path.join(outpath, "OverestVar.logFC.gct")
    else:
        sys.exit(
            "need to specify type as one of 'wilcox' or 't-test overest var'  or 't-test'"
        )

    # write out rankfile
    with open(gct_rank_File, "w") as fp:
        fp.write("#1.2" + "\n")
        fp.write(
            str(scores.shape[0]) + "\t" + str(scores.shape[1] - 1) + "\n"
        )  # "description" already merged in as a column
    fp.close()
    scores.to_csv(
        gct_rank_File, sep="\t", index=True, header=True, mode="a", float_format="%.3f"
    )
    print(gct_rank_File, "written out")

    # write out pvalues
    with open(gct_pvalue_File, "w") as fp:
        fp.write("#1.2" + "\n")
        fp.write(
            str(pvalues.shape[0]) + "\t" + str(pvalues.shape[1] - 1) + "\n"
        )  # "description" already merged in as a column
    fp.close()
    pvalues.to_csv(
        gct_pvalue_File,
        sep="\t",
        index=True,
        header=True,
        mode="a",
        float_format="%.3e",
    )
    print(gct_pvalue_File, "written out")

    # write out logFC
    with open(gct_logFC_File, "w") as fp:
        fp.write("#1.2" + "\n")
        fp.write(
            str(logFC.shape[0]) + "\t" + str(logFC.shape[1] - 1) + "\n"
        )  # "description" already merged in as a column
    fp.close()
    logFC.to_csv(
        gct_logFC_File, sep="\t", index=True, header=True, mode="a", float_format="%.3f"
    )
    print(gct_logFC_File, "written out")

    return None
    sys.exit(0)

    
## From Besca (https://github.com/bedapub/besca/blob/main/besca/st/_FAIR_export.py)
    
def export_metadata(adata, basepath, n_pcs=3, umap=True, tsne=False):
    """Export metadata in FAIR format for loading into database

    wrapper function for analysis_metadata with correct folder structure/names
    for loading into the database.

    parameters
    ----------
    adata: `AnnData`
        AnnData object that is to be exported
    basepath: `str`
        root path to the Analysis folder (i.e. ../analyzed/<ANALYSIS_NAME>)
    n_pcs: `int` | default = 3
        number of PCA components to export
    umap: `bool` | default = True
        boolian indicator if umap coordinates should be exported
    tsne: `bool` | default = False
        boolian indicator if tSNE coordinates should be exported

    returns
    -------
    None
        writes to file

    """

    analysis_metadata(adata, outpath=os.path.join(basepath), n_pcs=n_pcs, umap=umap, tsne=tsne)
    
    
## From Besca (https://github.com/bedapub/besca/blob/main/besca/export/_export.py)
    
def analysis_metadata(
    ## Sligh modification to keep the cells ID
    adata_input: AnnData,
    outpath: str = None,
    filename: str = "analysis_metadata.tsv",
    total_counts: bool = True,
    n_pcs: int = 3,
    umap: bool = True,
    tsne: bool = False,
    percent_mito: bool = True,
    n_genes: bool = True,
):
    """export plotting coordinates to analysis_metadata.tsv

    This function exports the indicated plotting coordinates or calculated PCAs to a .tsv file. This
    can be used to either transfer the data between different analysis platforms but can also be
    uploaded to the scseq database sicne the file follows the FAIR document format.

    To ensure FAIR compatibility the filename should not be changed.

    parameters
    ----------
    adata_input: `AnnData`
        the AnnData object containing the metadata
    outpath `str` | default = current working directory
        filepath to the directory in which the results should be outputed, if no directory is
        specified it outputs the results to the current working directory.
    filename: `str` | default = 'analysis_metadata.tsv'
        filename of the file that is to be written out
    total_counts: `bool` | default = True
        boolian indicator if total counts should be written out or not
    n_pcs: `int` | default = 3
        indicates number of PCA components that should be written out. If 0 no PCA components will be written out
    umap: `bool` | default = True
        boolian indicator if UMAP coordinates should be written out.
    tsne: `bool` | default = False
        boolian indicator if tSNE coordinates should be written out.
    percent_mito: `bool` | default = True
        boolian indicator if percent_mito should be written out or not
    n_genes: `bool` | default = True
        boolian indicator if n_genes should be written out or not

    returns
    -------
    None
        file is written out.

    """
    if outpath is None:
        outpath = os.getcwd()
    
    # add cellbarcodes to index
    adata = adata_input.copy()
   
    adata.obs = adata.obs.reset_index().set_index('CELL')
    data = pd.DataFrame(data=None, index=adata.obs.index) 

    if total_counts:
        if "n_counts" in adata.obs.columns:
            data["totalCounts"] = adata.obs.n_counts.copy()
        else:
            sys.exit(
                "need to have calculated 'n_counts' and stored it in adata.obs, consider running \"\""
            )

    if percent_mito:
        ## Slightly modify to make it work in Spatial. To Harmonize. 
        # if "percent_mito" in adata.obs.columns:
        #    data["percent_mito"] = adata.obs.percent_mito.copy()
        if "pct_counts_mt" in adata.obs.columns:
            data["percent_mito"] = adata.obs.pct_counts_mt.copy()
            
        else:
            print(
                "need to have calculated 'percent_mito' and stored it in adata.obs, percent mito will not be exported"
            )

    if n_genes:
        ## Slightly modify to make it work in Spatial. To Harmonize. 
        # if "n_genes" in adata.obs.columns:
        #    data["n_genes"] = adata.obs.n_genes.copy()
        if "n_genes_by_counts" in adata.obs.columns:
            data["n_genes"] = adata.obs.n_genes_by_counts.copy()
        else:
            print(
                "need to have calculated 'n_genes' and stored it in adata.obs, n_genes will not be exported"
            )

    obsm = adata.obsm.to_df()

    if n_pcs > 0:
        for i in range(1, n_pcs + 1):
            if obsm.get("X_pca" + str(i)) is None:
                sys.exit(
                    "number of PCA components requested not saved in adata.obsm, please ensure that PCA components have been calculated"
                )
            else:
                data["PCA.PC" + str(i)] = obsm.get("X_pca" + str(i)).tolist()

    if umap:
        if obsm.get("X_umap1") is None:
            sys.exit(
                "no UMAP coordinates found in adata.obsm, please ensure that UMAP coordinates have been calculated"
            )
        else:
            data["UMAP.c1"] = obsm.get("X_umap1").tolist()
            data["UMAP.c2"] = obsm.get("X_umap2").tolist()

    if tsne:
        if obsm.get("X_tsne1") is None:
            sys.exit(
                "no tSNE coordinates found in adata.obsm, please ensure that tSNE coordinates have been calculated"
            )
        else:
            data["tSNE.c1"] = obsm.get("X_tsne1").tolist()
            data["tSNE.c2"] = obsm.get("X_tsne2").tolist()

    ### check if the outdir exists if not create
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    data.to_csv(
        os.path.join(outpath, "analysis_metadata.tsv"),
        sep="\t",
        index_label="CELL",
        float_format="%.3f",
    )
    print("results successfully written out to 'analysis_metadata.tsv'")
    return None
    sys.exit(0)

    
### Function to export scores into the format required by MongoDB. 

def export_scores(
    path: Union[str, Path],
    scores: pd.DataFrame(),
    method_name: str='Chrysalis',
    number_decimals: int = 2,
    score_description: str= 'Tissue Compartments Score',
): 

    df_scores = scores.round(number_decimals)
    path_to_write = os.path.join(path, method_name)
    
    ### check if the outdir exists if not create
    if not os.path.exists(path_to_write):
        os.makedirs(path_to_write)
    
    df_scores.to_csv(os.path.join(path_to_write, "cell2scores.tsv"), sep='\t', index=True)
    print("results successfully written out to 'cell2scores.tsv'")
    
    scores_info(outpath = path_to_write, description = score_description, method = method_name)
    
    return None
    

# Adapted from Besca's labeling_info (https://github.com/bedapub/besca/blob/main/besca/export/_export.py)
    
def scores_info(
    outpath: str = None,
    description: str = None,
    public: bool = False,
    default: bool = True,
    expert: bool = False,
    reference: bool = False,
    method: str = None,
    annotated_version_of: str = "-",
    filename: str = "scoreinfo.tsv",
) -> None:
    """write out labeling info for uploading to database

    This functions outputs the file labelinfo.tsv which is needed to annotate a written out
    labeling in the scseq database.

    parameters
    ----------
    outpath: `str` | default = current working directory
        The filepath as a string indicating the location where the file should be written out to.
    description: `str` | 
        string describing what type of information is saved in the corresponding labeling.
    public: `bool` | default = False
        boolian indicator if the contained labeling information is available in the public domain.
    default_ `bool` | default = True
        boolian indicator if the labeling was created using a standardized process e.g. the leiden
        clusters outputed by the standard pipeline (this should be false if expert is true)
    expert: `bool` | default = False
        boolian indicator if the labeling was created by an 'expert' i.e. manually done (this should
        be false if default is true)
    reference: `bool` | default = True
        boolian indicator if this is the labeling (e.g. celltype annotation) that should be used for further analysis
        (there should only be one reference labeling per study)
    method: `str` |
        string indicating the type of method that was applied, e.g. if the labeling is of a clustering
        which clustering algorithm was used.
    annotated_version_of: `str` | default = '-'
        string identifying of what othe labeling/dataset this is an annotated version of (so for
        example if the labeling is celltype annotation of a leiden clustering then this would
        reference the leiden clsutering that was used to obtain the clusters that were then
        labeled here)
    filename: `str` | default = 'scoreinfo.tsv'
        string indicating the filename that should be used. This is per default set to the correct
        file name for uploading to the scseq database.

    returns
    -------
    None
        results are written out to a file instead

    """
    if outpath is None:
        outpath = os.getcwd()
    if public:
        Public = "TRUE"
    else:
        Public = "FALSE"

    if default:
        Default = "TRUE"
    else:
        Default = "FALSE"

    if expert:
        Expert = "TRUE"
    else:
        Expert = "FALSE"

    if reference:
        Reference = "TRUE"
    else:
        Reference = "FALSE"

    ciFile = os.path.join(outpath, filename)
    with open(ciFile, "w") as fp:
        fp.write(
            "description\tisPublic\tisDefault\tisExpert\tisReference\tmethod\tannotated_version_of\n"
        )
        fp.write(
            description
            + "\t"
            + Public
            + "\t"
            + Default
            + "\t"
            + Expert
            + "\t"
            + Reference
            + "\t"
            + method
            + "\t"
            + annotated_version_of
            + "\n"
        )
    fp.close()
    print(f"{filename} successfully written out")

    return None
    sys.exit(0)

##########################
# Wrapper to compute the spatially variable genes based on Chrysalis implementation
##########################

# See the documentation of Chrysalis for better understading of the parameters
# https://chrysalis.readthedocs.io/en/latest/generated/chrysalis.detect_svgs.html#chrysalis.detect_svgs


def compute_svg(
    list_adatas: list[AnnData],
    min_spots: Optional[int] = 0.05,
    top_svg: Optional[int] = 1000,
    min_morans: Optional[float] = 0.2,
    neighbors: Optional[int] = 6,
    sample_ID: Optional[str] = 'readout_id',
) -> list[AnnData]:

    adata_objects: list[AnnData] = []

    for current_adata in list_adatas:
        
        current_sample=current_adata.obs[sample_ID].unique().tolist()
        print(current_sample)
    
        ch.detect_svgs(current_adata, min_spots=min_spots, top_svg=top_svg, min_morans= min_morans, neighbors=neighbors)
    
        ch.plot_svgs(current_adata)
        plt.show()
        
        sc.pp.normalize_total(current_adata, inplace=True)
        sc.pp.log1p(current_adata)

        ch.pca(current_adata, n_pcs=50)
    
        ch.plot_explained_variance(current_adata)
        plt.show()
        
        adata_objects.append(current_adata)
        
    return adata_objects

##########################
# Wrapper to compute the tissue comparments based on archetipal analysis as implemented in Chrysalis. 
##########################

# See the documentation of Chrysalis for better understading of the parameters
# https://chrysalis.readthedocs.io/en/latest/generated/chrysalis.detect_svgs.html#chrysalis.detect_svgs

def compute_aa(
    list_adatas: list[AnnData],
    number_compartments: int = None,
    n_pcs: Optional[int] = 15,
    sample_ID: Optional[str] = 'readout_id',
    ncols: Optional[int] = 4,
) -> list[AnnData]:

    adata_objects: list[AnnData] = []

    for current_adata in list_adatas:
        
        current_sample=current_adata.obs[sample_ID].unique().tolist()
        print(current_sample)
        
        ch.aa(current_adata, n_pcs=n_pcs, n_archetypes=number_compartments)
        
        ch.plot(current_adata, dim=number_compartments)
        plt.show()
    
        ch.plot_heatmap(current_adata)
        plt.show()
        
        ch.plot_weights(current_adata)
        plt.show()
    
        ch.plot_compartments(current_adata, ncols=ncols)
        plt.show()
   
        adata_objects.append(current_adata)    
    
    return adata_objects