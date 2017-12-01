from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row,SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col, countDistinct, udf
from pyspark.mllib.linalg import Matrix, Vector, Vectors
from pyspark.mllib.linalg.distributed import RowMatrix, CoordinateMatrix, MatrixEntry
import pyspark as ps
from pyspark import SparkContext
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import json
sc = SparkContext(appName="SVDRecommender")
spark = ps.sql.SparkSession.builder.master("local[*]").appName('SVDRecommender').getOrCreate()


class PySparkRecommender(object):
    '''
    INSERT DOCSTRING HERE
    '''
    def __init__(self,filename):

        ratings = spark.read.option("inferSchema","true").option("header","true").csv(filename)
        # self.num_users, self.num_items, _, _= ratings.agg(*(countDistinct(col(c)).alias(c) for c in ratings.columns)).first()
        self.num_users, self.num_items = ratings.groupBy("userId").max(), ratings.groupBy("movieId").max()
        # msk = np.random.choice(np.arange(1,self.num_items+1), round(self.num_items*0.3), replace=False)
        function = udf(lambda c: c in msk, BooleanType())
        newdf = ratings.withColumn('test', function(col('movieId'))).drop(col('timestamp'))
        self.trainDF = newdf.filter(col('test') == 'False')
        self.testDF = newdf.filter(col('test') == 'True')
        aggTup = udf(lambda x,y: x.zip(y))
        rdd_tr = [self.trainDF.rdd.map(lambda r: Vectors.sparse(self.num_items, {r.movieId: r.rating}))]
        rdd_te = [self.testDF.rdd.map(lambda r: Vectors.sparse(self.num_items, {r.movieId: r.rating}))]
        # tr_rows = sc.parallelize(rdd_tr)
        # te_rows = sc.parallelize(rdd_te)
        # print(type(tr_rows))
        # tr_rowsBc = sc.broadcast(tr_rows)
        import ipdb; ipdb.set_trace()
        # te_rowsBc = sc.broadcast(te_rows)
        self.tr_mat = RowMatrix(rdd_tr)
        self.te_mat = RowMatrix(rdd_te)

    def trainSVD(self, k):
        svd = self.tr_mat.computeSVD(k,computeU=False)
        self.s_mat = svd.s
        self.k = k
        return self

    def testSVD(self):
        svd = self.te_mat.computeSVD(self.k,computeU=True)
        return svd.U, svd.V

    def test_model(self):
        pass


if __name__ == '__main__':
    psr = PySparkRecommender('data/movies/ratings.csv')
    psr.trainSVD(15)
