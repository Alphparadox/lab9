# part1_fire_analysis.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_date, year, month, weekofyear, count, desc, avg, round, expr
)
from pyspark.sql.types import DoubleType

def main():
    spark = SparkSession.builder \
        .appName("SF Fire 2018 Analysis") \
        .getOrCreate()

    # Path to your CSV file (change if needed)
    csv_path = "/opt/workdir/sf-fire-calls.csv"

    # Read CSV with header and infer schema (simple and flexible)
    df = spark.read.option("header", "true") \
                   .option("inferSchema", "true") \
                   .csv(csv_path)

    # Inspect schema briefly (uncomment for debugging)
    # df.printSchema()

    # Normalize date column (CallDate) -> convert to proper date
    # The dataset uses MM/dd/yyyy format in your example (e.g. 01/11/2002).
    df = df.withColumn("CallDate_parsed", to_date(col("CallDate"), "MM/dd/yyyy"))

    # Some datasets might have times inside CallDate or different format; if needed adjust format.
    # Filter only rows where CallDate_parsed is not null
    df = df.filter(col("CallDate_parsed").isNotNull())

    # Filter to year 2018
    df2018 = df.filter(year(col("CallDate_parsed")) == 2018).cache()
    print("Total rows for 2018:", df2018.count())

    # 1) All different types of fire calls in 2018
    call_types_2018 = df2018.select("CallType").distinct().orderBy("CallType")
    print("\n1) Different CallType values in 2018:")
    call_types_2018.show(truncate=False)

    # 2) Which months in 2018 saw the highest number of fire calls?
    calls_by_month = df2018.withColumn("month", month(col("CallDate_parsed"))) \
                           .groupBy("month") \
                           .agg(count("*").alias("num_calls")) \
                           .orderBy(desc("num_calls"))
    print("\n2) Number of calls by month (descending):")
    calls_by_month.show(12)

    # 3) Which neighborhood generated the most fire calls in 2018?
    calls_by_neighborhood = df2018.groupBy("Neighborhood") \
                                  .agg(count("*").alias("num_calls")) \
                                  .orderBy(desc("num_calls"))
    print("\n3) Top neighborhoods by number of calls:")
    calls_by_neighborhood.show(10, truncate=False)
    top_neighborhood = calls_by_neighborhood.limit(1).collect()
    if top_neighborhood:
        print("Neighborhood with most calls in 2018:", top_neighborhood[0]["Neighborhood"],
              "->", top_neighborhood[0]["num_calls"], "calls")

    # 4) Which neighborhoods had the worst response times to fire calls in 2018?
    # The dataset sample had a 'Delay' column (numeric). We'll compute average Delay per neighborhood.
    # Ensure Delay is numeric
    df2018 = df2018.withColumn("Delay_num", col("Delay").cast(DoubleType()))
    # If Delay has nulls, they will be ignored by avg
    avg_delay_by_neighborhood = df2018.groupBy("Neighborhood") \
                                      .agg(round(avg("Delay_num"), 2).alias("avg_delay")) \
                                      .filter(col("avg_delay").isNotNull()) \
                                      .orderBy(desc("avg_delay"))
    print("\n4) Neighborhoods with worst (highest) avg Delay (response time):")
    avg_delay_by_neighborhood.show(10, truncate=False)

    # 5) Which week in the year 2018 had the most fire calls?
    calls_by_week = df2018.withColumn("week", weekofyear(col("CallDate_parsed"))) \
                          .groupBy("week") \
                          .agg(count("*").alias("num_calls")) \
                          .orderBy(desc("num_calls"))
    print("\n5) Top weeks by number of calls:")
    calls_by_week.show(10)

    # 6) Is there a correlation between neighborhood, zip code, and number of fire calls?
    # Neighborhood is categorical — correlation is not directly meaningful. We'll show:
    #   a) calls per Zipcode
    #   b) calls per Neighborhood (already computed)
    #   c) Pearson correlation between zipcode (as integer) and calls-per-zipcode (this is approximate and only useful if zipcode is numeric)
    # Convert Zipcode to integer where possible
    df2018 = df2018.withColumn("Zipcode_num", expr("cast(Zipcode as int)"))
    calls_by_zip = df2018.groupBy("Zipcode_num").agg(count("*").alias("num_calls")).orderBy(desc("num_calls"))
    print("\n6a) Calls by Zipcode (top):")
    calls_by_zip.show(15)

    # Compute Pearson correlation between zipcode numeric and number of calls per zipcode (only where zipcode parsed)
    # Note: zipcode is categorical — numeric correlation may be meaningless; we print it with a warning.
    calls_by_zip_nonnull = calls_by_zip.filter(col("Zipcode_num").isNotNull())
    if calls_by_zip_nonnull.count() > 1:
        corr_val = calls_by_zip_nonnull.stat.corr("Zipcode_num", "num_calls")
        print("\nPearson correlation between Zipcode (numeric) and number of calls (per-zipcode):", corr_val)
        print("Warning: Zipcode is categorical. Numeric correlation may not be meaningful. Consider pivot tables or chi-square tests for categorical association.")
    else:
        print("\nNot enough zipcode data to compute correlation.")

    # Also show a pivot of top neighborhoods vs zipcodes (counts) to visually inspect relationship (top 20 neighborhoods)
    top_neighborhoods_list = [r["Neighborhood"] for r in calls_by_neighborhood.limit(20).collect()]
    print("\n6b) Pivot sample: counts of top neighborhoods per zipcode (limited output):")
    pivot_df = df2018.filter(col("Neighborhood").isin(top_neighborhoods_list)) \
                     .groupBy("Zipcode").pivot("Neighborhood").count()
    pivot_df.show(10, truncate=False)

    # 7) How to use Parquet files or SQL tables to store this data and read it back?
    # We'll save the filtered 2018 dataframe as Parquet and show how to read and create a temp SQL table.
    parquet_out = "/opt/workdir/output/sf_fire_2018_parquet"
    print(f"\n7) Writing 2018 data to Parquet at: {parquet_out}")
    df2018.write.mode("overwrite").parquet(parquet_out)

    print("Reading back Parquet into 'df_parquet' and creating a temp view 'sf_fire_2018'")
    df_parquet = spark.read.parquet(parquet_out)
    df_parquet.createOrReplaceTempView("sf_fire_2018")

    # Example Spark SQL queries
    print("\nExample SQL: Top 5 neighborhoods (from Parquet / SQL):")
    spark.sql("""
        SELECT Neighborhood, COUNT(*) AS num_calls
        FROM sf_fire_2018
        GROUP BY Neighborhood
        ORDER BY num_calls DESC
        LIMIT 5
    """).show(truncate=False)

    # Done
    print("\nAnalysis complete. Parquet file saved. You can load it later with spark.read.parquet(path).")
    spark.stop()

if __name__ == "__main__":
    main()
