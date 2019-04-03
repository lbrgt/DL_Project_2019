# DL_Project_2019

## Visualize the computation graph
- Download [Graphviz](http://www.graphviz.org/download/).
- Generate the computation graph by calling :
    ```Python
    import agtree2dot as ag 

    # ... compute loss of your network

    ag.save_dot(loss,{},open('./mlp.dot', 'w'))
    ```
    The graph is now stored in the file `mlp.dot`. 
 
- To render the computation graph, run the following command in your terminal:
    
    ```
    dot.exe mlp.dot -Lg -T pdf -o mlp.pdf
    ```
    
    `mlp.pdf` now contains the rendering of the computation graph. 

For more info, check https://fleuret.org/git/agtree2dot. 