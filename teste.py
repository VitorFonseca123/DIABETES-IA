import graphviz

dot_data = '''
digraph G {
    A -> B
    B -> C
    C -> A
}
'''

grafico = graphviz.Source(dot_data)
grafico.render(view=True)