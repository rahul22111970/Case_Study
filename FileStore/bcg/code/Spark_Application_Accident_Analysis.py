# Databricks notebook source
import requests
import json
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import *



spark = SparkSession.builder \
    .master("local") \
    .appName("Accident Analysis") \
    .getOrCreate()



print("Reading config file")

json_data = spark.read.json("dbfs:/FileStore/bcg/config/config.json",multiLine=True)

charges_path = json_data.select("Charges_use").head()[0]
damages_path = json_data.select("Damages_use").head()[0]
endorse_path = json_data.select("Endorse_use").head()[0]
person_path = json_data.select("Primary_Person_use").head()[0]
restrict_path = json_data.select("Restrict_use").head()[0]
units_path = json_data.select("Units_use").head()[0]
output_dir = json_data.select("Output_Dir").head()[0]



print("Reading data files")

charges_df = spark.read.format("csv").option("header","true").option("inferschema","true").load(charges_path)
damages_df = spark.read.format("csv").option("header","true").option("inferschema","true").load(damages_path)
endorse_df = spark.read.format("csv").option("header","true").option("inferschema","true").load(endorse_path)
person_df = spark.read.format("csv").option("header","true").option("inferschema","true").load(person_path)
restrict_df = spark.read.format("csv").option("header","true").option("inferschema","true").load(restrict_path)
units_df = spark.read.format("csv").option("header","true").option("inferschema","true").load(units_path)



print("Analyzing data")



#Analysis 1: Find the number of crashes (accidents) in which number of males killed are greater than 2

def analysis1():
    modified_person_df = person_df.filter(trim(col("prsn_gndr_id")) == "MALE").groupBy("crash_id").count().filter(col("count")>2)
    return modified_person_df.count()

output1 = analysis1()
print("Analysis 1 Output:")
print(output1)    



#Analysis 2: How many two wheelers are booked for crashes?

def analysis2():
    modified_units_df = units_df.filter(col("VEH_BODY_STYL_ID").like("%MOTORCYCLE%")).select("crash_id","unit_nbr").distinct()
    return modified_units_df.count()
output2 = analysis2()
print("Analysis 2 Output:")
print(output2)



#Analysis 3: Determine the Top 5 Vehicle Makes of the cars present in the crashes in which driver died and Airbags did not deploy

def analysis3():
    person_df1 = person_df.filter((trim(col("prsn_type_id")) == "DRIVER") & (trim(col("prsn_airbag_id")) == "NOT DEPLOYED") & (trim(col("PRSN_INJRY_SEV_ID")) == "KILLED"))
    
    units_df2 = units_df.filter(col("veh_body_styl_id").like('%CAR%'))

    join_df = person_df1.join(units_df2, (person_df1["crash_id"] == units_df2["crash_id"]) & (person_df1["unit_nbr"] == units_df2["unit_nbr"]), "inner").select(units_df2["crash_id"],units_df2["unit_nbr"],units_df2["veh_make_id"]).distinct()
    
    final_df = join_df.groupBy("veh_make_id").count().orderBy("count",ascending=False).limit(5)

    return final_df

output3 = analysis3()

print("Analysis 3 Output:")
output3.show()



#Analysis 4: Determine number of Vehicles with driver having valid licences involved in hit and run?

def analysis4():
    person_df2 = person_df.filter( (trim(col("prsn_type_id")).isin('DRIVER','DRIVER OF MOTORCYCLE TYPE VEHICLE')) & (trim(col("drvr_lic_type_id")).isin('DRIVER LICENSE','COMMERCIAL DRIVER LIC.')) )
    
    modified_charges_df = charges_df.filter((col("charge").like('%HIT AND RUN%')) | (col("charge").like('%HIT & RUN%')))
    
    return person_df2.join(modified_charges_df, (person_df2["crash_id"] == modified_charges_df["crash_id"]) & (person_df2["unit_nbr"] == modified_charges_df["unit_nbr"])).count()

output4 = analysis4()

print("Analysis 4 Output:")
print(output4)



#Analysis 5: Which state has highest number of accidents in which females are not involved

def analysis5():
    person_df3 = person_df.filter(~col("prsn_gndr_id").isin("FEMALE")).select("crash_id","drvr_lic_state_id").distinct().groupBy("drvr_lic_state_id").count().orderBy(col("count"),ascending=False).select("drvr_lic_state_id")

    return person_df3.first()["drvr_lic_state_id"]

output5 = analysis5()

print("Analysis 5 Output:")
print(output5)

# COMMAND ----------

#Analysis 6: Which are the Top 3rd to 5th VEH_MAKE_IDs that contribute to a largest number of injuries including death

def analysis6():
    person_df4 = person_df.select("crash_id","unit_nbr",expr("tot_injry_cnt+death_cnt").alias("inj_cnt"))
    
    units_df1 = units_df.select("crash_id","unit_nbr","veh_make_id").distinct()
    
    join_df = person_df4.join(units_df1,(person_df4["crash_id"] == units_df["crash_id"]) & (person_df4["unit_nbr"] == units_df["unit_nbr"]),"inner")
    
    mod_join_df = join_df.groupBy("veh_make_id").agg(expr("sum(inj_cnt)").alias("inj_sum"))
    
    windowSp = Window.orderBy(desc("inj_sum"))
    
    final_df = mod_join_df.withColumn("rnk", rank().over(windowSp)).filter((col("rnk")>=3) & (col("rnk")<=5)).select("veh_make_id")
    
    return final_df

output6 = analysis6()

print("Analysis 6 Output:")
output6.show()



#Analysis 7: For all the body styles involved in crashes, mention the top ethnic user group of each unique body style

def analysis7():
    units_bs_df = units_df.select("crash_id","unit_nbr","veh_body_styl_id").distinct()
    
    join_df = person_df.join(units_bs_df, (person_df["crash_id"] == units_bs_df["crash_id"]) & (person_df["unit_nbr"] == units_bs_df["unit_nbr"]),"inner")
    
    grouped_df = join_df.groupBy("veh_body_styl_id","prsn_ethnicity_id").count()
    
    windowSp = Window.partitionBy("veh_body_styl_id").orderBy(desc(col("count")))
    
    final_df = grouped_df.withColumn("rnk", rank().over(windowSp)).filter(col("rnk") == 1).select("veh_body_styl_id","prsn_ethnicity_id")
    
    return final_df

output7 = analysis7()

print("Analysis 7 Output:")
output7.show()



#Analysis 8: Among the crashed cars, what are the Top 5 Zip Codes with highest number crashes with alcohols as the contributing factor to a crash (Use Driver Zip Code)

def analysis8():
    units_al_df = units_df.select("crash_id","unit_nbr","veh_body_styl_id","contrib_factr_1_id").filter((col("veh_body_styl_id").like('%CAR%')) & ((trim(col("contrib_factr_1_id")) == 'UNDER INFLUENCE - ALCOHOL') | (trim(col("contrib_factr_2_id")) == 'UNDER INFLUENCE - ALCOHOL'))).distinct()
    
    person_df5=person_df.withColumnRenamed("CRASH_ID","PERSON_CRASH_ID")
    
    join_df = person_df5.join(units_al_df, (person_df5["person_crash_id"] == units_al_df["crash_id"]) & (person_df5["unit_nbr"] == units_al_df["unit_nbr"]),"inner")
    
    final_df = join_df.groupBy("drvr_zip").agg(countDistinct(col("person_crash_id")).alias("count")).orderBy(col("count"),ascending=False).filter(col("drvr_zip").isNotNull()).limit(5)
    
    return final_df

output8 = analysis8()

print("Analysis 8 Output:")
output8.show()



#Analysis 9: Count of Distinct Crash IDs where No Damaged Property was observed and Damage Level (VEH_DMAG_SCL~) is above 4 and car avails Insurance

def analysis9():
    units_df5 = units_df.filter(((col("veh_dmag_scl_1_id").isin('DAMAGED 5','DAMAGED 6','DAMAGED 7 HIGHEST')) | (col("veh_dmag_scl_2_id").isin('DAMAGED 5','DAMAGED 6','DAMAGED 7 HIGHEST'))) & (col("FIN_RESP_TYPE_ID").like('%INSURANCE%')) & (col("veh_body_styl_id").like('%CAR%'))).withColumnRenamed("CRASH_ID","UNITS_CRASH_ID")
    
    mod_damages_df = damages_df.filter(~trim(col("damaged_property")).isin("NONE","NONE1"))
    join_df = units_df5.join(mod_damages_df, units_df5["units_crash_id"] == mod_damages_df["crash_id"],"left_anti")

    damages_none_df = damages_df.filter(trim(col("damaged_property")).isin("NONE","NONE1")).select("crash_id").distinct()
    
    return join_df.select("UNITS_CRASH_ID").union(damages_none_df).distinct().count()

output9 = analysis9()

print("Analysis 9 Output:")
print(output9)



#Analysis 10: Determine the Top 5 Vehicle Makes where drivers are charged with speeding related offences, has licensed Drivers, used top 10 used vehicle colours and has car licensed with the Top 25 states with highest number of offences (to be deduced from the data)

def analysis10():
    top_states = units_df.select("crash_id","veh_lic_state_id").distinct().groupBy("veh_lic_state_id").count().orderBy("count",ascending=False).limit(25).select("veh_lic_state_id").withColumnRenamed("veh_lic_state_id","top_veh_lic_state_id")
    
    top_colous = units_df.select("crash_id","unit_nbr","veh_color_id").distinct().groupBy("veh_color_id").count().orderBy(col("count"),ascending=False).filter(col("veh_color_id") != 'NA').limit(10).select("veh_color_id")
    
    trans_units_df = units_df.select("crash_id","unit_nbr","veh_body_styl_id","veh_color_id","veh_lic_state_id","veh_make_id").distinct().withColumnRenamed("crash_id","units_crash_id").withColumnRenamed("unit_nbr","unit_own_nbr").filter(col("veh_body_styl_id").like('%CAR%')).join(top_states, units_df["veh_lic_state_id"] == top_states["top_veh_lic_state_id"])
    
    trans_units_df2 = trans_units_df.join(top_colous, trans_units_df["veh_color_id"] == top_colous["veh_color_id"])
    
    lic_drivers = person_df.filter( (trim(col("prsn_type_id")).isin('DRIVER','DRIVER OF MOTORCYCLE TYPE VEHICLE')) & (trim(col("drvr_lic_type_id")).isin('DRIVER LICENSE','COMMERCIAL DRIVER LIC.')) )
    
    trans_units_df3 = trans_units_df2.join(lic_drivers, (trans_units_df2["units_crash_id"] == lic_drivers["crash_id"]) & (trans_units_df2["unit_own_nbr"] == lic_drivers["unit_nbr"]))
    
    speed_df = charges_df.filter(col("charge").like('%SPEED%')).select("crash_id","unit_nbr").distinct()
    
    trans_units_df4 = trans_units_df3.join(speed_df, (trans_units_df3["units_crash_id"] == speed_df["crash_id"]) & (trans_units_df3["unit_own_nbr"] == speed_df["unit_nbr"]))
    
    final_df = trans_units_df4.select("unit_own_nbr","units_crash_id","veh_make_id").groupBy("veh_make_id").count().orderBy(col("count"),ascending=False).limit(5)
    
    return final_df

output10 = analysis10()

print("Analysis 10 Output:")
output10.show()


print("Writing data for ANALYSIS 1")

spark.createDataFrame([output1], StringType()).withColumnRenamed("value","output1").write.mode("overwrite").option("header","true").csv(output_dir+"/output1.csv")

print("Writing data for ANALYSIS 2")
spark.createDataFrame([output2], StringType()).withColumnRenamed("value","output2").write.mode("overwrite").option("header","true").csv(output_dir+"/output2.csv")

print("Writing data for ANALYSIS 3")
output3.write.mode("overwrite").option("header","true").csv(output_dir+"/output3.csv")

print("Writing data for ANALYSIS 4")
spark.createDataFrame([output4], StringType()).withColumnRenamed("value","output4").write.mode("overwrite").option("header","true").csv(output_dir+"/output4.csv")

print("Writing data for ANALYSIS 5")
spark.createDataFrame([output5], StringType()).withColumnRenamed("value","output5").write.mode("overwrite").option("header","true").csv(output_dir+"/output5.csv")

print("Writing data for ANALYSIS 6")
output6.write.mode("overwrite").option("header","true").csv(output_dir+"/output6.csv")

print("Writing data for ANALYSIS 7")
output7.write.mode("overwrite").option("header","true").csv(output_dir+"/output7.csv")

print("Writing data for ANALYSIS 8")
output8.write.mode("overwrite").option("header","true").csv(output_dir+"/output8.csv")

print("Writing data for ANALYSIS 9")
spark.createDataFrame([output9], IntegerType()).withColumnRenamed("value","output9").write.mode("overwrite").option("header","true").csv(output_dir+"/output9.csv")

print("Writing data for ANALYSIS 10")
output10.write.mode("overwrite").option("header","true").csv(output_dir+"/output10.csv")

print("Data Writing Complete")

spark.stop()




