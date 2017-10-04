from __future__ import print_function

"""
created by xywang 20-jan-2017
hierarchical co-clustering = bipartite coclustering + kmeans(spark builtin) and sim_ahc (in pyspark on mini-cluster)
script source: pyspark_coclustering.ipynb

This is a Spark implementation of Bipartite Co-clustering algorithm (ref: Dhillon 2001):
    1 given a doc-term matrix A, we obtain A'= D1_{-1/2} x A x D2_{-1/2}
    2 apply SVD on A' to obtain two sets of singular vectors U and V
    3 remove the 1st column of U and the 1st row of V
    4 compute D1_{-1/2}xU and D2_{-1/2}xV and combine their result to be Z
    5 apply a clustering method on Z to obtain coclusters.
This script implements steps 1-4, for step 5, we have "HierCo_Kmeans.py" and "HierCo_simHClust.py"

Input: a .mtx file from folder "/home/hduser/Documents/test_Spark/input_mtx"
        this .mtx file is output of preprocessing, it is a sparse doc-term matrix
output: a Z vector, as it is a result of BlockMatrix multiplication, it is parallelized by default,
        it is saved as a Spark pickle directory on HDFS, named as collection_name_z_par.
        Note that:
            - spark pickle file has to be on HDFS if the program is run on top of a cluster,
              if not, saving will have problem. If spark runs locally, Spark pickle will be saved locally.
            - For running a program on cluster, if the pickle file alreay exists on HDFS,
              it has to be removed first before re-saved again. HDFS cannot replace or update the
              new version automatically.

Some notable details:
    - pyspark does NOT have computSVD() method, a java wrapper is used.
      SVD() outputs U,s and V. They are of different types. U is a Spark distributed IndexedRowMatrix,
      and V is a Spark local dense matrix.
    - java computeSVD() also implements the "arpack" method, but different from sklearn.utils.arpack.svds,
      the singular values in s vector of computeSVD() are ordered decreasingly, like [1, 0.9, 0.8, ...].
      However, in the singular values in s vector of sklearn.utils.arpack.svds are ordered increasingly, like
      [..., 0.8, 0.9, 1]. In either case, the singular vectors in U and in V that match the largest singular value
      should be removed. So that, in Python co-clustering, the first column in U and V should be removed. And in
      the Spark program, the first column in U and V should be removed. But in Spark, it is easier to remove the first
      columns from D_1^{-1/2}XU and from D_2^{-1/2}XV.
    - In the end, columns of Z vectors output by the Python co-clustering are in the reverse order from the columns of
      Z vectors output by the Spark co-clustering. This should not effect the clustering result using K-means or AHC.
    - For Spark distributed matrices, multiply() function for the IndexedRowMatrix and for the RowMatrix functions different
      from the multiply() function of the BlockMatrix. For the IndexedRowMatrix and RowMatrix, their multiply() function only
      allows a distributed matrix to be multiplied by a Spark local dense matrix (on the right side). However,
      in our program, the right matrix is a large matrix, which is difficult to convert into a Spark local dense matrix.
      That is why we use BlockMatrix's multiply() function, which allows multiplcation of two BlockMatrices. In this way,
      we do not have to convert a scipy sparse matrix into a Spark local dense matrix. However, we do need to convert an IndexedRowMatrix
      to a BlockMatrix and vice versa. But I think it is a better way than the other option.
    - Input A is local .mtx file, D1_{-1/2} and D2_{-1/2} are generated from A as scipy.csr diagonal matrices.
      D1_{-1/2}xA is computed locally before the result is distributed by function _indexed_sp_vecs(), the restul
      eventually will be converted into a spark distributed BlockMatrix.
    - D2_{-1/2} is distributed by function _indexed_sp_vecs() and eventually converted into a spark distributed BlockMatrix.
    - D1_{-1/2}xAxD2_{-1/2} is obtained by multiplcation of two BlockMatrix, but it has to be converted to be an
      IndexedRowMatrix so that it can be used as input in computeSVD() function.
    - D1_{-1/2} and U are converted to BlockMatrix to do multiplication, then to IndexedRowMatrix, then to an RDD,
      in which the 1st column will be removed by removing the 1st value in each value vector.
    - V is converted to a local numpy array, and D2_{-1/2}xV is done locally, and the first value in each vector is removed.
      The result is sc.parallelized again in order to combine the result of D1_{-1/2}xU to produce Z.

modified on 16/Feb/2017:
      As the singular values in sigma are ordered decreasingly in computSVD() function,
      the 1st column in U and in V are removed accordingly, so that singular vectors of the 2nd largest singular
      values and below are kept. This is based on the bipartite spectral paper.
      Experiments show that doing so, ARI with ground-truth is improved for Reuter_10C_2450D collection.
re-verified on 05/Sep/2017:
      Codes' comments are corrected. Results are re-verified.
"""

import os, sys, pickle
os.environ['SPARK_HOME']="/home/hduser/spark-2.0.0-bin-hadoop2.7"
sys.path.append("/home/hduser/spark-2.0.0-bin-hadoop2.7/python")

import numpy as np
from time import time
from scipy import io
from scipy.sparse import diags
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.mllib.linalg import SparseVector, Matrices, _convert_to_vector, DenseMatrix
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix, BlockMatrix
from pyspark.mllib.common import callMLlibFunc, JavaModelWrapper

class SVD(JavaModelWrapper):
    """
    Wrapper around the SVD scala case class
    type(U) = Spark distributed IndexedRowMatrid or RowMatrix
    type(s) = Spark local DenseVector
    type(V) = Spark local DenseMatrix
    """
    @property
    def U(self):
        """ Returns a RowMatrix whose columns are the left singular vectors of the SVD if computeU was set to be True."""
        u = self.call("U")
        if u is not None:
            mat_name = u.getClass().getSimpleName()
            if mat_name == "RowMatrix":
                return RowMatrix(u)
            elif mat_name == "IndexedRowMatrix":
                return IndexedRowMatrix(u)
            else:
                raise TypeError("Expected RowMatrix/IndexedRowMatrix got %s" % mat_name)

    @property
    def s(self):
        """Returns a DenseVector with singular values in descending order."""
        return self.call("s")

    @property
    def V(self):
        """ Returns a DenseMatrix whose columns are the right singular vectors of the SVD."""
        return self.call("V")

def computeSVD(row_matrix, k, computeU=False, rCond=1e-9):

    java_model = row_matrix._java_matrix_wrapper.call("computeSVD", int(k), computeU, float(rCond))
    return SVD(java_model)

if __name__ == "__main__":

    """ config Spark and create a SparkContext """

    conf = SparkConf().\
        setMaster("spark://159.84.139.244:7077").\
        set("spark.executor.memory", "6g").\
        setAppName("HCoClust")

    """ num of cores will only be in effect when app is launched via spark-submit
    ./spark-submit \
    --master spark://159.84.139.244:7077 \
    --executor-memory 6G \
    --total-executor-cores 5 \
    /home/hduser/Documents/test_Spark/HierCoClust.py
    """

    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    sc.setCheckpointDir("hdfs://master:9000/RddCheckPoint")

    """ input user parameters """

    f_name = "AP_21000d_doc_term.mtx"
    f_n = f_name.replace("_doc_term.mtx", "")
    K = 3
    num_cores = 10

    """applying SVD on normalized A' """

    A = io.mmread("/home/hduser/Documents/test_Spark" + "/" + "input_mtx" + "/" + f_name).tocsr() #os.getcwd()
    d1_sqrt = np.asarray(1.0 / np.sqrt(A.sum(axis=1)))  # d1_sqrt = D1^(-1/2)
    ls_ = d1_sqrt.tolist()  # a list of lists, each list contains a value
    d1_sqrt_ls = [val for subls in ls_ for val in subls]    # a list of d1_sqrt values
    d1_sqrt_n = len(d1_sqrt_ls)
    d1_sqrt_mat = diags(d1_sqrt_ls, offsets=0, shape=(d1_sqrt_n, d1_sqrt_n), format='csr')
    d1_A = d1_sqrt_mat * A

    def _indexed_sp_vecs(sp_mat):
        """
        input: a matrix in csr format
        output: a list of indexed spark sparse vectors
        """
        num_r, num_c = sp_mat.shape
        ind_ = [i for i in range(num_r)]
        data_ls = map(lambda d: d.data, sp_mat)
        ind_ls = map(lambda d: d.indices, sp_mat)
        ind_val = map(lambda d1, d2: dict(zip(d1, d2)), ind_ls, data_ls)
        indexed_sp_vecs_ = map(lambda ind, fts: (ind, SparseVector(num_c, fts)), ind_, ind_val)
        return indexed_sp_vecs_

    d1_A_indexed_sp_vecs = _indexed_sp_vecs(d1_A)
    IRM_d1_A = IndexedRowMatrix(sc.parallelize(d1_A_indexed_sp_vecs).cache())

    d2_sqrt = np.asarray(1.0 / np.sqrt(A.sum(axis=0)))
    _ls = d2_sqrt.tolist()
    d2_sqrt_ls = [val for subls in _ls for val in subls] # a list of d1_sqrt values
    d2_sqrt_n = len(d2_sqrt_ls)

    # convert IRM_d1_A and d2_sqrt_dense to BlockMatrices, and do multiplication
    d2_sqrt_mat = diags(d2_sqrt_ls, offsets=0, shape=(d2_sqrt_n, d2_sqrt_n), format='csr')
    d2_indexed_sp_vecs = _indexed_sp_vecs(d2_sqrt_mat)
    IRM_d2 = IndexedRowMatrix(sc.parallelize(d2_indexed_sp_vecs, num_cores))
    BM_d2 = IRM_d2.toBlockMatrix()

    BM_d1_A = IRM_d1_A.toBlockMatrix()
    BM_d1_A_d2 = BM_d1_A.multiply(BM_d2)
    IRM_d1_A_d2 = BM_d1_A_d2.toIndexedRowMatrix()

    n_sv = 1 + int(np.ceil(np.log2(K)))
    t0 = time()
    svd = computeSVD(IRM_d1_A_d2, n_sv, True)
    print(time()-t0)

    """compute D1^{-1/2} \times U"""

    IRM_d1 = IndexedRowMatrix(sc.parallelize(_indexed_sp_vecs(d1_sqrt_mat)))    # construct a Spark IndexedRowMatrix from d1_sqrt_mat
    BM_d1 = IRM_d1.toBlockMatrix()  # then convert this IndexedRowMatrix into a BlockMatrix
    U = svd.U
    # test = U.rows.collect()
    # test_s = svd.s
    BM_U = U.toBlockMatrix()    # convert the IndexedRowMatrix U into a BlockMatrix
    BM_d1_U = BM_d1.multiply(BM_U)  # do BlockMatrix multiplication
    IRM_d1_U = BM_d1_U.toIndexedRowMatrix() # convert the resulted BlockMatrix into an IndexedRowMatrix

    """convert the IndexedRowMatrix into a dataframe, then to a common RDD.
       as singular values output by computeSVD() are in descending order,
       so we need to remove the first column"""

    IRM_d1_U_rows_df = callMLlibFunc("getIndexedRows", IRM_d1_U._java_matrix_wrapper._java_model)
    d1_U_ind_par = IRM_d1_U_rows_df.rdd.map(lambda row: (row[0], row[1][1:]))

    """
    - construct a sparse diagonal matrix from d2_sqrt_ls
    - convert Spark dense matrix (a local matrix) svd.V to a np.matrix
    - and do multiplication
    """

    V = svd.V
    V_array = V.toArray()   # convert V to array using DenseMatrix.toArray() method
    V_mat = np.matrix(V_array)  # convert into V_mat so to do mulplication
    d2_V = d2_sqrt_mat * V_mat
    total_n = d1_sqrt_n + d2_sqrt_n

    print(d1_sqrt_n, d2_sqrt_n, total_n)

    ind_ = [i for i in range(d1_sqrt_n, total_n)]
    d2_V_ind = map(lambda ind, vec: (ind, vec[1:]), ind_, np.array(d2_V))   # remove the first col of V = remove the first row of Vt
    d2_V_ind_par = sc.parallelize(d2_V_ind) # parallelize the indexed vectors d2_V_ind

    z = d1_U_ind_par + d2_V_ind_par
    # test_z = z_par.take(20)

    save_path = "hdfs://master:9000/user/xywang/CH/"
    z.saveAsPickleFile(save_path + f_n + "_z_par")

    sc.stop()

"""
    # have to be sure that the _z_par file is not on hdfs, following code doesn' work, need to execute manually
    # if os.system("hadoop fs -test -d /user/xywang/" + f_n + "_z_par") == 0:
    #     print("The z_par file already exists on hdfs. It will be removed ...")
    #     os.system("hadoop fs -rm -r /user/xywang/" + f_n + "_z_par")   # remove the z_par file
    # else:
    #     print("The z_par file has been saved on hdfs.")

    # def _multiply(indexed_rowmatrix, matrix):
    #     not in use, only useful for small local dataset
    #     if not isinstance(matrix, DenseMatrix):
    #         raise ValueError("Only multiplication with DenseMatrix is supported.")
    #     return IndexedRowMatrix(indexed_rowmatrix._java_matrix_wrapper.call("multiply", matrix))
"""
