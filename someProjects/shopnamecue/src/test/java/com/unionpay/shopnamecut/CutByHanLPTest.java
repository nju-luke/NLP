package com.unionpay.shopnamecut;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;

/**
 * Created by luke on 16-8-2.
 */
@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(locations = {"classpath:spring/shopnamecut-spring.xml"})
public class CutByHanLPTest {
    @Autowired
    CutByHanLP cutByHanLP;

    @Test
    public void testLineToWords() throws Exception {
        System.out.println("1");
        String line = "今天是个好日子，温度终于降下来了";
        System.out.println("2");
        System.out.println(cutByHanLP.lineToWords(line));
        System.out.println("3");
    }


}