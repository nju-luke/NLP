import org.ansj.domain.Result;
import org.ansj.domain.Term;
import org.ansj.splitWord.analysis.NlpAnalysis;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by luke on 16-7-31.
 */
public class WikiCutByAnsj {
    public String filePath;

    public WikiCutByAnsj(String filePath) {
        this.filePath = filePath;
    }

    public WikiCutByAnsj() {

    }

    public void cut(String filePath) throws IOException {
        /*File file = new File(filePath);
        InputStreamReader read = new InputStreamReader(
                new FileInputStream(file));
        BufferedReader br = new BufferedReader(read);*/

        BufferedReader br = fileToBr(filePath);
        BufferedWriter bw = newFileBr(filePath);
        String line = null;
        List<String> list = new ArrayList<>();
        while ((line = br.readLine()) != null) {
            if(isPassed(line)) continue;
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
        if(line.startsWith("<")) flag = true;
        if(line.length() < 40) flag = true;
        return flag;
    }

    public BufferedReader fileToBr(String filePath) throws IOException {
        File file = new File(filePath);
        InputStreamReader read = new InputStreamReader(
                new FileInputStream(file));
        BufferedReader br = new BufferedReader(read);
        return br;
    }

    public BufferedWriter newFileBr(String filePath) throws IOException {
        File file = new File(filePath + "_cut");
        FileWriter writer = new FileWriter(file);
        return (BufferedWriter) new BufferedWriter(writer);
    }


//    public void wikiCut(String filePath){
//
//    }

/*    public String readLine() {

    }*/

    public List<String> lineToWords(String line) {
        Result resut = NlpAnalysis.parse(line);
        List<String> list = new ArrayList<>();
        for (Term term : resut
                ) {
            list.add(term.getName());
        }
        return list;
    }


}
