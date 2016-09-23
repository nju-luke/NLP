//package com.unionpay.shopnamecut;
//
//import org.springframework.stereotype.Component;
//
//import java.util.ArrayList;
//import java.util.Collections;
//import java.util.List;
//import java.util.Random;
//
///**
// * Created by luke on 16-8-2.
// */
//@Component
//public class GeographicCoordinates_backup {
//
//    private static final double MAXRadius = Math.sqrt(82400 / Math.PI);
//    private static final double DIGREETODISTENCE = 111;
//    private static final double RATIAO = 0.8;
//    private static final int treeNumbers = 10;
//
////    public double[] calCenterCoordinates(List<double[]> coordinatesList) {
////        double[][] pooledCoordinatesArray = pool2Array(coordinatesList);                            //忘了把最小值加回来了，囧囧囧囧囧囧囧囧囧囧囧囧
////        double[] centerCoordinates = randomForest(pooledCoordinatesArray, treeNumbers);
////        return centerCoordinates;
////    }
//
//    public double[] calCenterCoordinates(List<double[]> coordinatesList) {
//        double[][] pooledCoordinatesArray = pool2Array(coordinatesList);
//        double[] centerCoordinates = removeDeviation(pooledCoordinatesArray);
//
//        return centerCoordinates;
//    }
//
//    private double[] removeDeviation(double[][] pooledCoordinatesArray) {
//        double[] centiTudes = chopedMeanArray(pooledCoordinatesArray);
//        return new double[0];
//    }
//
//    private double[] randomForest(double[][] pooledCoordinatesArray, int treeNumbers) {     //随机选取比例为RATIAO的数据产生中心，最后通过随机中心平均，或者去除偏离点再求平均
//        int lengthOfArray = pooledCoordinatesArray[0].length;
//        int choseSize = (int) ((int) treeNumbers * RATIAO);
//        List<Integer> indexList = new ArrayList<>();
//
//        for (int i = 0; i < lengthOfArray; i++) {
//            indexList.add(i);
//        }
//        double[][] centitudes = new double[treeNumbers][];
//        for (int i = 0; i < treeNumbers; i++) {
//            centitudes[i] = randomCenterArray(pooledCoordinatesArray, indexList, choseSize, lengthOfArray);
//        }
//
//        double[] returnArray;
////        returnArray = meanArray(centitudes);                                                 //直接取中心平均
//        returnArray = chopedMeanArray(centitudes);                                          //去除偏离点取中心平均
//        returnArray[0] /= DIGREETODISTENCE;
//        returnArray[1] /= DIGREETODISTENCE;
//
//        return returnArray;
//    }
//
//    public double[] meanArray(double[][] array) {                           //只给定array则直接求出第二维的平均，给定indexList时则求第二维对应位置的平均
//        double sumCenterLongitude = 0.0;
//        double sumCenterLatitude = 0.0;
//        int arraySize = array[0].length;
//        for (int i = 0; i < arraySize; i++) {
//            sumCenterLongitude += array[i][0];
//            sumCenterLatitude += array[i][1];
//        }
//        double[] mean = new double[0];
//        mean[0] = sumCenterLongitude / arraySize;
//        mean[1] = sumCenterLatitude / arraySize;
//        return mean;
//    }
//
//    public double[] meanArray(double[][] array, List<Integer> indexList) {
//        double sumCenterLongitude = 0.0;
//        double sumCenterLatitude = 0.0;
//        for (int i : indexList) {
//            sumCenterLongitude += array[i][0];
//            sumCenterLatitude += array[i][1];
//        }
//        double[] mean = new double[2];
//        mean[0] = sumCenterLongitude / indexList.size();
//        mean[1] = sumCenterLatitude / indexList.size();
//        return mean;
//    }
//
//    public double[] chopedMeanArray(double[][] array) {                                     //删除（1-RATIAO）的偏离坐标再求中心
//        int arraySize = array[0].length;
//        List<Double> distences = new ArrayList<>();
//        List<Integer> indexList = new ArrayList<>();
//        for (int i = 0; i < arraySize; i++) {
//            distences.add(relativeDistences(array, i));                  //求出所有相对距离之和
//            indexList.add(i);
//        }
//        double maxDistence = 0.0;
//        for (int i = 0; i < arraySize * (1 - RATIAO); i++) {                     //删除（1-RATIAO）比例的偏差较远的坐标
//            maxDistence = Collections.max(distences);
//            indexList.remove(distences.indexOf(maxDistence));
//            distences.remove(distences.indexOf(maxDistence));
//        }
//        double[] returnArray;
//        returnArray = meanArray(array, indexList);
//        return returnArray;
//    }
//
//    public double relativeDistences(double[][] array, int index) {
//        double sumDistences = 0.0;
//        for (int i = 0; i < array[0].length; i++) {
//            sumDistences += Math.sqrt(Math.pow(array[0][i] - array[0][index], 2) + Math.pow(array[1][i] - array[1][index], 2));
//        }
//        return sumDistences;
//    }
//
//    public double[] randomCenterArray(double[][] pooledCoordinatesArray, List<Integer> indexList, int choseSize, int lengthOfArray) {
//        Random random = new Random();                                           //随机有放回的抽取choseSize个数据求中心坐标
//        Collections.shuffle(indexList);
//        Double sumLongitude = 0.0;
//        Double sumLatitude = 0.0;
//        for (int j = 0; j < choseSize; j++) {
//            int index = random.nextInt(choseSize);
//            sumLongitude += pooledCoordinatesArray[0][indexList.indexOf(index)];
//            sumLatitude += pooledCoordinatesArray[1][indexList.indexOf(index)];
//        }
//        double centitude[] = new double[2];
//        centitude[0] = sumLongitude / choseSize;
//        centitude[1] = sumLatitude / choseSize;
//
//        return centitude;
//    }
//
//    public double[][] pool2Array(List<double[]> coordinatesList) {                  //基准化坐标值
//        List<Double> longitudes = new ArrayList<>();
//        List<Double> latitudes = new ArrayList<>();
//        for (double[] coordinates : coordinatesList) {
//            longitudes.add(coordinates[0] * DIGREETODISTENCE);
//            latitudes.add(coordinates[1] * DIGREETODISTENCE);
//        }
//        double[][] coorrdinates = new double[2][];
//        coorrdinates[0] = poolBase(longitudes);
//        coorrdinates[1] = poolBase(latitudes);
//        return coorrdinates;
//    }
//
//    public double[] poolBase(List<Double> xtudes) {
//        double minTude = Collections.min(xtudes);
//        double[] tempArray = new double[xtudes.size()];
//        int i = 0;
//        for (double num :
//                xtudes) {
//            tempArray[i++] = num;
//        }
//        for (i = 0; i < xtudes.size(); i++) {
//            tempArray[i] -= minTude;
//        }
//        return tempArray;
//    }
//}
