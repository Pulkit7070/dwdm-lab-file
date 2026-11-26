echo "Name: Pulkit Saraf | Enrollment No: 10617702723"
java -cp weka.jar weka.core.Instances weather.numeric.arff
java -cp weka.jar weka.filters.unsupervised.attribute.Discretize -B 4 -R first-last -i weather.numeric.arff -o weather_discretized.arff
java -cp weka.jar weka.core.Instances weather_discretized.arff
