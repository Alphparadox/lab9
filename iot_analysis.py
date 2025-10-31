from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import os, shutil, traceback

spark = SparkSession.builder.appName("IoT_Analysis").getOrCreate()

IN_JSON = "/opt/workdir/iot_devices.json"
TMP_OUT = "/tmp/output/iot"
FINAL_OUT = "/opt/workdir/output/iot"

df = spark.read.json(IN_JSON)
print("IoT schema:")
df.printSchema()
print("Row count:", df.count())

threshold = 2
failing = df.filter(F.col("battery_level").cast("double") < threshold).select("device_id","device_name","cn","battery_level")
print(f"\n1) Devices with battery_level < {threshold}:")
failing.show(200, truncate=False)

co2_threshold = 1000
high_co2 = df.filter(F.col("c02_level").cast("double") > co2_threshold).groupBy("cn").agg(F.count("*").alias("num_devices"), F.round(F.avg("c02_level"),2).alias("avg_c02")).orderBy(F.desc("avg_c02"))
print(f"\n2) Countries with devices having CO2 > {co2_threshold}:")
high_co2.show(100, truncate=False)

metrics = df.agg(
    F.min("temp").alias("min_temp"),
    F.max("temp").alias("max_temp"),
    F.min("battery_level").alias("min_battery"),
    F.max("battery_level").alias("max_battery"),
    F.min("c02_level").alias("min_co2"),
    F.max("c02_level").alias("max_co2"),
    F.min("humidity").alias("min_humidity"),
    F.max("humidity").alias("max_humidity"),
)
print("\n3) Min/Max metrics:")
metrics.show(truncate=False)

avg_temp_by_country = df.groupBy("cn").agg(F.round(F.avg("temp"),2).alias("avg_temp"), F.count("*").alias("num_devices")).orderBy(F.desc("avg_temp"))
print("\n4) Average temperature by country (descending):")
avg_temp_by_country.show(100, truncate=False)

print("\nWriting results to tmp and copying to mounted folder if possible...")
os.makedirs(TMP_OUT, exist_ok=True)
try:
    shutil.rmtree(TMP_OUT, ignore_errors=True)
except:
    pass
os.makedirs(TMP_OUT, exist_ok=True)

df.write.mode("overwrite").parquet(os.path.join(TMP_OUT, "iot_full"))
failing.write.mode("overwrite").parquet(os.path.join(TMP_OUT, "failing_devices"))
high_co2.write.mode("overwrite").parquet(os.path.join(TMP_OUT, "high_co2"))
metrics.write.mode("overwrite").parquet(os.path.join(TMP_OUT, "min_max"))
avg_temp_by_country.write.mode("overwrite").parquet(os.path.join(TMP_OUT, "avg_temp_by_country"))

print("Saved parquet to", TMP_OUT)

try:
    if os.path.exists(FINAL_OUT):
        shutil.rmtree(FINAL_OUT)
    shutil.copytree(TMP_OUT, FINAL_OUT)
    print("Copied output to", FINAL_OUT)
except Exception as e:
    print("Could not copy to mounted folder automatically. Use docker cp to copy /tmp to host:")
    print(f"  docker cp spark-master:{TMP_OUT} C:\\Users\\ASUS\\lab8\\output")
    traceback.print_exc()

spark.stop()
print("IoT analysis completed.")
