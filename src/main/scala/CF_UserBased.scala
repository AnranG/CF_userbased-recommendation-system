    import java.io.{File, PrintWriter}
    import breeze.numerics.{abs, sqrt}
    import org.apache.spark.{SparkConf, SparkContext}
    import org.apache.spark.storage.StorageLevel
    import scala.collection.Set



    object CF_UserBased {

      def main(args: Array[String]) {
        val t0 = System.nanoTime()



        val conf = new SparkConf()
        conf.setAppName("CF_UserBased")
        conf.setMaster("local[*]")
        conf.set("spark.executor.memory", "1g")
        conf.set("driver-memory", "4g")
        conf.set("executor-cores", "2")


        val sc = new SparkContext(conf)
        val storageLevel = StorageLevel.MEMORY_ONLY


        var total_file = sc.textFile("data/video_small_num.csv").cache()
          .map(line => line.split(","))
        val total_header = total_file.first()
        total_file = total_file.filter(_ (0) != total_header(0))

        val total = total_file.map(line => (line(0).toInt, (line(1).toInt, line(2).toDouble)))


        var testing_file = sc.textFile("data/video_small_testing_num.csv").cache().map(line => line.split(","))
        val testing_header = testing_file.first()
        testing_file = testing_file.filter(_ (0) != testing_header(0))

        val testing_rdd = testing_file.map(line => (line(0).toInt, (line(1).toInt, line(2).toDouble)))


        val training_rdd = total.subtract(testing_rdd) //(uID,(pID,rating))
          .groupByKey().mapValues(value => value.toList)


        //calc mean and normalize

        val rdd_avg = training_rdd.map { case (uID, pro_rating_list) =>

          val sum = pro_rating_list.reduce((p1, p2) => (1, p1._2 + p2._2))

          val avg = sum._2 / pro_rating_list.length

          (uID, (avg, pro_rating_list))
        }


        val not_zero = rdd_avg
          //        .filter { case((uID,avg), pro_rating_list) =>
          //       var all_zero = true
          //
          //        pro_rating_list.toMap.valuesIterator.toList.size !=1
          //      }
          .map { case (uID, (avg, pro_rating_list)) => (uID, (avg, pro_rating_list.toMap)) }


        val upr_map = not_zero.collect().toMap
        upr_map.foreach(println)

        val user_product_set_rdd = not_zero.map { case (uID, (avg, pro_rating_map)) =>

          val product_set = pro_rating_map.keysIterator.to[scala.collection.Set]

          (uID, (avg, product_set))
        }



        //calc co-rating avg

        val user_product_set = user_product_set_rdd.collect().sortBy(_._1)
        // user_product_set.foreach(println)


        val uu_pearson_map = calc_pearson(user_product_set,upr_map)


       // uu_pearson_map.foreach(println)
      //  println(uu_pearson_map.size)



        // predict

        val testing_data = testing_rdd.collect()

        val testing_wPrediction = testing_data.map { case (userId_A, (the_proId, og_rating)) =>

          val corate_up = user_product_set.filter { case (userId_B, (avg, pro_rating_list)) =>
            Set(the_proId).subsetOf(pro_rating_list.toSet)
          }// now corate_upr_map contains only product-rating only when uerB also have rated pro before


          val avg_A = upr_map(userId_A)._1
          var predict_numerator = 0.0
          var predict_denominator = 0.0
          corate_up.map{case (userId_B, (avg_B, pro_rating_list)) =>

              //to find pearson between userA ans userB
            var pearson = 0.0
            if(uu_pearson_map.contains(Set(userId_A,userId_B))){
              pearson = uu_pearson_map(Set(userId_A,userId_B))
            }

            if(pearson!=0) {


              val this_rating =  (upr_map(userId_B))._2(the_proId)
              val avg_B_new_total =(upr_map(userId_B))._2.valuesIterator.toList.sum-this_rating
              val avg_B_new_num = (upr_map(userId_B))._2.size-1
              val avg_B_new = avg_B_new_total/avg_B_new_num


              predict_numerator += (this_rating-avg_B_new)*pearson
              predict_denominator += abs(pearson)

            }

          }

          var predict =0.0
          if(predict_numerator*predict_denominator==0) {
             predict = avg_A

          }else{
            predict = avg_A + predict_numerator / predict_denominator
          }

        ((userId_A,the_proId),(og_rating,predict))
        }.sortBy(x=>(x._1._1,x._1._2))
        testing_wPrediction.foreach(println)





        val diff= testing_wPrediction.map { case ((user, product), (r1, r2)) => math.abs(r1 - r2)}

        var num1=0
        var num2=0
        var num3=0
        var num4=0
        var num5=0
        for ( x <- diff) {
          x match {
            case x if (x>=0 && x<1) => num1+=1;
            case x if (x>=1 && x<2)=> num2+=1;
            case x if (x>=2 && x<3) => num3+=1;
            case x if (x>=3 && x<4) => num4+=1;
            case x if (x>=4 ) => num5+=1;
          }
        }

        println(">=0 and <1:"+ num1)
        println(">=1 and <2:"+ num2)
        println(">=2 and <3:"+ num3)
        println(">=3 and <4:"+ num4)
        println(">=4 :"+ num5)


        var MSE = 0.0

        val calc_MSE = testing_wPrediction.map { case ((user, product), (r1, r2)) =>
          MSE += scala.math.pow(r1-r2,2)
        }
        val RMSE = sqrt(MSE/testing_wPrediction.length)

        println("RMSE:"+ RMSE)


        val output_file = new File("data/UserBasedCF_prediction.txt")
        val out = new PrintWriter(output_file)
        for (elem <- testing_wPrediction){

          out.write(elem._1._1 + "," + elem._1._2 +","+ elem._2._2 +"\n")

        }

        out.close()


        val t1 = System.nanoTime()
        println("Elapsed time: " + (t1 - t0) + "ns")

      }

      def calc_pearson(user_product_set:Array[(Int,(Double,Set[Int]))],upr_map:Map[Int,(Double,Map[Int,Double])]):
      Map[Set[Int], Double] ={
        var uu_pearson_map = Map.empty[Set[Int], Double]

        for (i <- 0 until user_product_set.length) {
          for (j <- i + 1 until user_product_set.length) {

            val product_intersect = user_product_set(i)._2._2.intersect(user_product_set(j)._2._2)


            if (product_intersect.size > 6) {

              var sum_i = 0.0
              var sum_j = 0.0

              for (elem <- product_intersect) {

                val i_map = upr_map(user_product_set(i)._1)._2
                val rating_i = i_map(elem)


                sum_i += rating_i.toDouble

                val j_map = upr_map(user_product_set(j)._1)._2
                val rating_j = j_map(elem)
                sum_j += rating_j.toDouble


              }

              val co_avg_i = sum_i / product_intersect.size
              val co_avg_j = sum_j / product_intersect.size

              var numerator = 0.0
              var denominator_i = 0.0
              var denominator_j = 0.0


              for (elem <- product_intersect) {

                val i_map = upr_map(user_product_set(i)._1)._2
                val rating_i = i_map(elem)
                val j_map = upr_map(user_product_set(j)._1)._2
                val rating_j = j_map(elem)

                numerator += (rating_i - co_avg_i) * (rating_j - co_avg_j)
                denominator_i += scala.math.pow(rating_i - co_avg_i, 2)
                denominator_j += scala.math.pow(rating_j - co_avg_j, 2)

              }
              //var weight = 0.0
              if (denominator_i * denominator_j * numerator != 0) {
                val weight = numerator / sqrt(denominator_i * denominator_j)
                uu_pearson_map += (Set(user_product_set(i)._1, user_product_set(j)._1) -> weight)
              }
            }

          }


        } // end of ixj loop
        return uu_pearson_map
      }

        }//object






