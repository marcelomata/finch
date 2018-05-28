Translation from Chinese to English is done by [Zhedong Zheng](https://github.com/zhedongzheng)

---

* 第一讲 知识图谱概览

  Chapter 1 - Knowledge Graph Overview

    * KG的本质
    
      Essence of KG

        * Web视角: 像建立文本之间的超链接一样，建立数据之间的语义链接，并支持语义搜索

          Web: similar to hyperlinks between text, build semantic links between data, and support semantic searching

        * NLP视角: 怎样从文本中抽取语义和结构化数据

          NLP: how to extract semantic and structural data from text

        * KR视角: 怎样利用计算机符号来表示和处理知识

          KR: how to use computational symbols to represent and process knowledge

        * AI视角: 怎样利用知识库来辅助理解人的语言

          AI: how to use knowledge base to help understand human language

        * DB视角: 用图的方式去存储知识

          DB: store knowledge in graphs

---

* 第二讲 知识表示和知识建模

  Chapter 2 - Knowledge Representation and modelling
  
    * RDF
        * RDF的意思是 资源描述框架（Resource Description Framework）
        
          RDF represents Resource Description Framework
  
        * 在RDF中，知识总是以三元组形式出现 (主，谓，宾)

          In RDF, knowledge appear as triple (subject, predicate, object)
          
        * RDF同样是一种图模型，将对资源的描述连接在一起
        
          RDF is also a graphical model to link the descriptions of resources
          
          RDF三元组可以被认为是图上的有方向单元（结点，箭头，结点）
          
          RDF triples can be seen as arcs of a graph (vertex, edge, vertex)
 
    * OWL
