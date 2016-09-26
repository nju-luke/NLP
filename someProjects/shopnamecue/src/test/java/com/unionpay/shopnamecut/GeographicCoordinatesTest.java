package com.unionpay.shopnamecut;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by luke on 16-8-3.
 */

@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(locations = {"classpath:spring/shopnamecut-spring.xml"})
public class GeographicCoordinatesTest {
    @Autowired
    GeographicCoordinates geographicCoordinates;

//    @Autowired

    @Test
    public void testClass() {
        List<double[]> coordinatesList = new ArrayList<>() ;
        Random random = new Random();

//        for (int i = 0; i < 100; i++) {
//            double digree = 30;
//            if (random.nextInt(10)>8) digree = 180*random.nextDouble();
//            double[] cordinates = new double[2];
//            cordinates[0] = random.nextDouble()*1+digree;
//            cordinates[1]  = random.nextDouble()*1+digree;
//            coordinatesList.add(cordinates);
////            System.out.println(cordinates[0]+","+cordinates[1]);
//        }

//        System.out.println(Arrays.toString(coordinatesList.get(3)));
        double[] temp = new double[2];

        temp = new double[]{34.452219, 87.890626};
        coordinatesList.add(temp);

        temp = new double[]{31.883065, 93.78652599999999};
        coordinatesList.add(temp);

        temp = new double[]{33.18898, 88.83866999999999};
        coordinatesList.add(temp);

        temp = new double[]{31.88327, 94};
        coordinatesList.add(temp);

        temp = new double[]{35.74651, 87.01170999999999};
        coordinatesList.add(temp);

        double[] cordinates = new double[2];
        cordinates = geographicCoordinates.calCenterCoordinates(coordinatesList);
        System.out.println(cordinates[0]+","+cordinates[1]);

    }
}