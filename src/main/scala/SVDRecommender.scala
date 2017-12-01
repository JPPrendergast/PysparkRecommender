import org.apache.spark.mllib.linalg.{ Matrix, SingularValueDecomposition, Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix

import org.apache.spark.ml.clustering.LDA
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, countDistinct, udf}
import org.apache.spark.{SparkConf,SparkContext}

object SVDRecommender {

  def main(args: Array[A]): Unit = {
    // creates sparksession
    val conf = new SparkConf().setAppName("SVDRecommender")
    // val sc = new SparkContext(conf)
    val spark = new SparkSession.master("local").builder.appName(s"${this.getClass.getSimpleName}").config("spark.sql.warehouse.dir", "./spark-warehouse/").getOrCreate()
    // spark.start()
    // Trains a LDA Model
    val ratings = spark.read.option("inferSchema","true").option("header","true").csv("data/movies/ratings.csv")
    // val ratings_pivot = movies.groupBy("userId").pivot("movieId").agg(expr("coalesce(first(rating),3)").cast("double"))
    // val ratings_rdd: RDD[Row] = ratings_pivot.rdd
    // val rat: = ratings_rdd.map(x => Vector(x))
    // val rows = sc.parallelize(rat)
    val nums = ratings.agg(c => _.countDistinct(col(c)).alias(c))).first()
    val num_users = nums.userId
    val num_items = nums.movieId
    def splitMask(f: Double)(n: Double): Int = {math round (n*f)}

    val msk = for (i <- 1 to splitMask(0.3)(num_items)) yield scala.util.Random.nextInt(num_items)
    val filterFunction = udf(Row: Boolean => Boolean = msk.contains(_))
    val newDF = ratings.withColumn("test", filterFunction('movieId))
    val trainDF = newDF.filter('test == false)
    val trainDF = newDF.filter('test == true)
    val trainRows = sc.parallelize(trainDF.rdd.map(r => Vectors.sparse(r.userId, {r.movieId, r.rating}))))
    val mat: RowMatrix = new RowMatrix(rows)
    val svd: SingularValueDecomposition[RowMatrix,Matrix] = mat.computeSVD(35, 5, computeU=true)
    val U: RowMatrix = svd.U
    val s: Vector = svd.s
    val V: Matrix = svd.V

    val model = lda.fit(movies)

    val ll = model.logLikelihood(movies)
    val lp = model.logPerplexity(movies)
    println(s"The lower bound on the log likelihood of the entire corpus: $ll")
    println(s"The upper bound on perplexity: $lp")

    // Describe topics.
    val topics = model.describeTopics(3)
    println(s"The topics described by their top-weighted terms: $topics")
    topics.show(true)

    val transformed = model.transform(movies)
    transformed.show(false)
    spark.stop()
  }
}

import org.apache.spark.sql.Row
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types.{ArrayType, IntegerType, DoubleType, DataType, StructType, StructField}

class TupleAggregation extends UserDefinedAggregateFunction {
  override def inputSchema: StructType = StructType(
    StructField("movieId", IntegerType, false  ) ::
    StructField("rating" , DoubleType , false  ) :: Nil
  )
  override def bufferSchema: StructType = StructType(
    ArrayType(
      StructType(
        Tuple2(
          StructField("a", IntegerType),StructField("b", DoubleType)
        )
      )
    ) :: Nil
  )
  override def dataType: DataType = ArrayType(
    StructType(
      Tuple2(
        StructField(
          "a",
          IntegerType
        ),
        StructField(
          "b",
          DoubleType
        )
      )
    )
  )
  override def deterministic: Boolean = true
  override def initialize(buffer: MutableAggregationBuffer): Unit = {
    buffer(0) = IndexedSeq[Tuple2[Int, Double]]()
  }
  override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    if (buffer != null){
      val seq = buffer(0).asInstanceOf[IndexedSeq[Tuple2[Int,Double]]]
      buffer(0) = seq :+ Tuple2(input.getAs[Int](0), input.getAs[Double](1))
    }
  }
  override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
    if (buffer1(0) != null && buffer2 != null){
      val seq1 = buffer1(0).asInstanceOf[IndexedSeq[Tuple2[Int,Double]]]
      val seq2 = buffer2(0).asInstanceOf[IndexedSeq[Tuple2[Int,Double]]]
      buffer1(0) = seq1 ++ seq2
    }
  }
  override def evaluate(buffer: Row): Any = {
    buffer(0).asInstanceOf[IndexedSeq[Tuple2[Int,Double]]]
  }
}










import org.apache.spark.sql.Row
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types.{ArrayType, LongType, DataType, StructType, StructField}

class CollectionFunction(private val limit: Int) extends UserDefinedAggregateFunction {
    def inputSchema: StructType =
        StructType(StructField("movieId", IntegerType, false) ::
                   StructField("rating" , DoubleType , false  ) :: Nil)

    def bufferSchema: StructType =
      StructType(StructField("list", ArrayType(Tuple2(LongType,DoubleType), true), true) :: Nil)

    override def dataType: DataType = ArrayType(LongType, true)

    def deterministic: Boolean = true

    def initialize(buffer: MutableAggregationBuffer): Unit = {
        buffer(0) = IndexedSeq[Long]()
    }

    def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
        if (buffer != null) {
            val seq = buffer(0).asInstanceOf[IndexedSeq[Long]]
            if (seq.length < limit) {
                buffer(0) = input.getAs[Long](0) +: seq
            } else {
                buffer(0) = null
            }
        }
    }

    def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
        if (buffer1(0) != null && buffer2 != null) {
            val seq1 = buffer1(0).asInstanceOf[IndexedSeq[Long]]
            val seq2 = buffer2(0).asInstanceOf[IndexedSeq[Long]]
            if (seq1.length + seq2.length <= limit) {
                buffer1(0) = seq1 ++ seq2
            } else {
                buffer1(0) = null
            }
        }
    }

    def evaluate(buffer: Row): Any = {
        if (buffer(0) == null) {
            IndexedSeq[Long]()
        } else {
            buffer(0).asInstanceOf[IndexedSeq[Long]]
        }
    }
}

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}
val conf = new SparkConf().setAppName("agg")
val sc = new SparkContext(conf)
val sqlContext = new SQLContext(sc)
val a = sc.parallelize(0L to 20L).map(x => (x, x % 4)).toDF("value", "group")
val cl = new CollectionFunction(5)
val df = a.groupBy("group").agg(cl($"value").as("list")).cache()

import org.apache.spark.sql.{Encoder, Encoders}
case class Data(i: Int, j: Double)
val function = new Aggregator[Data, List[(Int, Double)], List[(Int, Double)]] {
  def bufferEncoder: Encoder[List[(Int,Double)]] = Any
  def outputEncoder: Encoder = List[Encoders.tuple[Encoders.scalaInt, Encoders.scalaDouble]]
  def zero: List[(Int, Double)] = List()
  def reduce(b: List[(Int, Double)], a: Data): List[(Int, Double)] = b :+ (a.i, a.j)
  def merge(b1: List[(Int, Double)], b2: List[(Int, Double)]): List[(Int, Double)] = b1 ++ b2
  def finish(r: List[(Int, Double)]): List[(Int, Double)] = r
}.toColumn()


package org.apache.spark.sql

import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.expressions.Aggregator
import org.apache.spark.sql.expressions.scalalang.typed
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StringType
import org.apache.spark.sql.{Encoder, Encoders}

case class AggData(a: Int, b: Double)
object SeqAgg extends Aggregator[AggData, Seq[Int], Seq[(Int, Int)]] {
  def zero: Seq[Int] = Nil
  def reduce(b: Seq[Int], a: AggData): Seq[Int] = a.a +: b
  def merge(b1: Seq[Int], b2: Seq[Int]): Seq[Int] = b1 ++ b2
  def finish(r: Seq[Int]): Seq[(Int, Int)] = r.map(i => i -> i)
  override def bufferEncoder: Encoder[Seq[Int]] = ExpressionEncoder()
  override def outputEncoder: Encoder[Seq[(Int, Int)]] = ExpressionEncoder()
}

  // def bufferEncoder: Encoder[List[(Int,Double)]] = List(Encoders.tuple(Encoders.scalaInt,Encoders.scalaDouble))
  // def outputEncoder: Encoder[List[(Int,Double)]] = Encoders.tuple(Encoders.scalaInt,Encoders.scalaDouble)
