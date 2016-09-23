package com.unionpay.shopnamecut;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.seg.common.Term;
import org.springframework.stereotype.Component;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by luke on 16-8-2.
 */

@Component
public class CutByHanLP {

//    private String filePath;

//    public CutByHanLP(String filePath) {
//        this.filePath = filePath;
//    }

    public void cut(String filePath) throws IOException {
        /*File file = new File(filePath);
        InputStreamReader read = new InputStreamReader(
                new FileInputStream(file));
        BufferedReader br = new BufferedReader(read);*/

        BufferedReader br = fileToBr(filePath);
        BufferedWriter bw = newFileBr(filePath);
        String line;
        List<String> list;
        while ((line = br.readLine()) != null) {
            if (isPassed(line)) continue;
            list = lineToWords(line);
            for (String word :
                    list) {
                bw.write(word + " ");
            }
            bw.write("\n");
        }
        br.close();
        bw.close();
    }

    private boolean isPassed(String line) {
        boolean flag = false;
        if (line.length() <= 2) flag = true;
        return flag;
    }

    public BufferedReader fileToBr(String filePath) throws IOException {
        File file = new File(filePath);
        InputStreamReader read = new InputStreamReader(
                new FileInputStream(file));
        return new BufferedReader(read);
    }

    public BufferedWriter newFileBr(String filePath) throws IOException {
        File file = new File(filePath + "_cut");
        FileWriter writer = new FileWriter(file);
        return new BufferedWriter(writer);
    }

    public List<String> lineToWords(String line) {
//        Segment segment = new CRFSegment();
//        List<Term> terms = segment.seg(line);
        List<Term> termList = HanLP.segment(line);

//        Result resut = parse(line);
        List<String> list = new ArrayList<>();
        for (Term term : termList
                ) {
            list.add(term.word);
        }
        return list;
    }

}

