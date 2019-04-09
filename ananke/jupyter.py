import ipywidgets as widgets
import plotly.graph_objs as go
from plotly.offline import iplot
import numpy as np

def cluster_size_explorer(dbs):
    @widgets.interact_manual(feature_id=
                            widgets.Dropdown(
                            options=sort_index,
                            description="Feature:"))
    def plot_nseqs_vs_eps(feature_id):
        nclusts = []
        for eps in dbs.dist_range:
            try:
                nclusts.append(len(dbs.DBSCAN(eps, expand_around=feature_id, max_members=200, warn=False)[1]))
            except:
                break
        layout = dict(title = 'Number of Sequences in Cluster %d vs. Radius (Epsilon)' %(feature_id,),
                      xaxis = dict(title = 'Epsilon'),
                      yaxis = dict(title = 'Number of Sequences'),
                      )
        fig = {"data": [go.Scatter(x=dbs.dist_range, y=nclusts)],
               "layout": layout}
        iplot(fig)
    
    return plot_nseqs_vs_eps

def snapshot(adb, dbs, tax_index):
    abundances = np.zeros(shape=(adb.nts,), dtype=np.int)
    for series in adb.get_series():
        for i in range(adb.nts):
            abundances[i] += sum(adb[i, series])
            sort_index = np.argsort(abundances)
    display(widgets.Text(disabled=True, 
                             value="Data Partitioning, Feature Drop-down contains all sequences not included in a snapshotted cluster",
                            layout=widgets.Layout(width="600px")))
    w_feat = widgets.Dropdown(
        options=[str(x) + ": " + tax_index[adb.featureids[x].decode()] for x in sort_index[-2000:]],
        value=str(sort_index[-1]) + ": " + tax_index[adb.featureids[sort_index[-1]].decode()],
        description='Feature:',
        disabled=False
    )
    w_eps=widgets.Dropdown(
        options=dbs.dist_range,
        description='Epsilon:')
    
    w_legend = widgets.Checkbox(
        value=True,
        description="Show Legend (in case of issues)")
    
    w_members = widgets.IntText(value=200, description="Max Plot Size:")
    
    w_snapshots = widgets.Select(options=[], disabled=True, description="Snapshots:")
    
    session_snapshots = []
    
    sequence_snapshots = set()
    
    def plot_clusters(feature_choice, eps, show_legend=True, max_members=200, snapshots=[]):
        feature_index = int(feature_choice.split(": ")[0])
        x = adb.plot_feature_at_distance(dbs, int(feature_index), eps, title_index=tax_index, max_members=max_members)
        
        if x:
            x['layout']['showlegend'] = show_legend
            iplot(x)
            button = widgets.Button(description="Snapshot Cluster")
            notes = widgets.Textarea(description="Notes")
            display(notes, button)
    
            def on_button_clicked(b):
                for seq_id in x['data']:
                    seq_id = int(seq_id['name'].split(":")[0])
                    sequence_snapshots.add(seq_id)
                opts=[str(x) + ": " + tax_index[adb.featureids[x].decode()] for x in sort_index[-2000:] if x not in sequence_snapshots]
                w_feat.options = opts
                w_feat.value = opts[-1]
                session_snapshots.append((feature_index, eps, notes.value))
                w_snapshots.options = session_snapshots
                print("Snapshot of cluster %s saved!" % (feature_index,))
    
            button.on_click(on_button_clicked)
    
    
    return widgets.interact_manual(plot_clusters, feature_choice=w_feat, 
                            eps=w_eps, show_legend=w_legend, max_members=w_members,
                            snapshots=w_snapshots)
