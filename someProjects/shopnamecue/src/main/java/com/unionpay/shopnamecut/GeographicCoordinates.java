package com.unionpay.shopnamecut;

import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by luke on 16-8-2.
 */
@Component
public class GeographicCoordinates {

    private static final double MAXRadius = Math.sqrt(82400 / Math.PI);
    private static final double DIGREETODISTENCE = 111;
    private static final double RATIAO = 0.8;


    public double[] calCenterCoordinates(List<double[]> coordinatesList) {
        double[][] coordinatesArray = corrdinates2Array(coordinatesList);
        double[] centerCoordinates = removeDeviation(coordinatesArray);

        return centerCoordinates;
    }

    private double[] removeDeviation(double[][] coordinatesArray) {
        double[] centiTudes = chopedMeanArray(coordinatesArray);
        double[] distences2Center = distences2Center(coordinatesArray, centiTudes);

        System.out.println(MAXRadius);
        double sumCenterLongitude = centiTudes[0];
        double sumCenterLatitude = centiTudes[1];
        int length = 1;
        for (int i = 0; i < distences2Center.length; i++) {
            if (distences2Center[i] > MAXRadius) continue;
            sumCenterLongitude += coordinatesArray[0][i];
            sumCenterLatitude += coordinatesArray[1][i];
            length++;
        }
        double[] mean = new double[2];
        mean[0] = sumCenterLongitude / length / DIGREETODISTENCE;
        mean[1] = sumCenterLatitude / length / DIGREETODISTENCE;
        return mean;
    }

    private double[] distences2Center(double[][] coordinatesArray, double[] centiTudes) {
        double[] distences = new double[coordinatesArray[0].length];
        for (int i = 0; i < distences.length; i++) {
            distences[i] = Math.sqrt(Math.pow(coordinatesArray[0][i] - centiTudes[0], 2) + Math.pow(coordinatesArray[1][i] - centiTudes[1], 2));
        }
        return distences;
    }

    public double[] meanArray(double[][] array, List<Integer> indexList) {
        double sumCenterLongitude = 0.0;
        double sumCenterLatitude = 0.0;
        for (int i : indexList) {
            sumCenterLongitude += array[0][i];
            sumCenterLatitude += array[1][i];
        }
        double[] mean = new double[2];
        mean[0] = sumCenterLongitude / indexList.size();
        mean[1] = sumCenterLatitude / indexList.size();
        return mean;
    }

    public double[] chopedMeanArray(double[][] array) {                                     //删除（1-RATIAO）的偏离坐标再求中心
        int arraySize = array[0].length;
        List<Double> distences = new ArrayList<>();
        List<Integer> indexList = new ArrayList<>();
        for (int i = 0; i < arraySize; i++) {
            distences.add(relativeDistences(array, i));                  //求出所有相对距离之和
            indexList.add(i);
        }
        double maxDistence = 0.0;
        for (int i = 0; i < arraySize * (1 - RATIAO); i++) {                     //删除（1-RATIAO）比例的偏差较远的坐标
            maxDistence = Collections.max(distences);
            indexList.remove(distences.indexOf(maxDistence));
            distences.remove(distences.indexOf(maxDistence));
        }
        double[] returnArray;
        returnArray = meanArray(array, indexList);
        return returnArray;
    }

    public double relativeDistences(double[][] array, int index) {
        double sumDistences = 0.0;
        for (int i = 0; i < array[0].length; i++) {
            sumDistences += Math.sqrt(Math.pow(array[0][i] - array[0][index], 2) + Math.pow(array[1][i] - array[1][index], 2));
        }
        return sumDistences;
    }

    public double[][] corrdinates2Array(List<double[]> coordinatesList) {                  //基准化坐标值
        List<Double> longitudes = new ArrayList<>();
        List<Double> latitudes = new ArrayList<>();
        for (double[] coordinates : coordinatesList) {
            longitudes.add(coordinates[0] * DIGREETODISTENCE);
            latitudes.add(coordinates[1] * DIGREETODISTENCE);
        }

        double[][] coordinates = new double[2][longitudes.size()];
        for (int i = 0; i < longitudes.size(); i++) {
            coordinates[0][i] = longitudes.get(i);
            coordinates[1][i] = latitudes.get(i);
        }
        return coordinates;
    }

}
