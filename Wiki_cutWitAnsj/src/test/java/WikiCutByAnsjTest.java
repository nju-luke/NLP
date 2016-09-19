import org.junit.Test;

import java.io.IOException;
import java.util.List;

/**
 * Created by luke on 16-8-1.
 */
public class WikiCutByAnsjTest {
    WikiCutByAnsj wcb = new WikiCutByAnsj();

    @Test
    public void lineToWords() throws Exception {

        String line = "细心的朋友还会发现，在project-01目录下，新生成了一个target（项目输出）文件夹，下面包括surefire-reports（测试结果）和编译过后的class文件。mvn test可以很好的支持单元测试，maven下的好多命令可以完成其中奇葩怪异的任务，并且mvn 命令支持串行执行。比如，mvn  install、mvn clean build等等。";
        List<String> list = wcb.lineToWords(line);
        System.out.println(list);
    }

    @Test
    public void testCut() throws IOException {
        String filePath = "/media/luke/工作/Wiki/wiki00_chs";
        wcb.cut(filePath);
    }

}