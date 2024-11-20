import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re

def get_event_memory_performance(dom):
    #mem_data = glob.glob('bc_training/*memory_size*.csv')
    mem_data = [f for f in os.listdir('bc_training_logs/') if re.search('.*[0-9]+.csv', f)]
    dfs = []
    d = dom.replace(" ", "")
    for fname in mem_data:
        m = re.search('ep\_data\_[0-9]+\_([a-zA-Z]+\-v[0-9])\_memory\_size\_[0-9]+.csv', fname)
        if m:
            domain = m.group(1)
            if d.lower() in domain.lower():
                df = pd.read_csv(f'bc_training_logs/{fname}')#.iloc[-1:]
                dfs.append(df)
    mem_df = pd.concat(dfs).reset_index(drop=True)#.sort_values('Num Trajectories')
    mem_df = mem_df.astype({"Domain": str, "Num Schemas": int, "Num Events": int, "Observation": int, "Num Trajectories": int})
    mem_df = mem_df.dropna(axis='columns')
    #mem_df = mem_df.drop_duplicates()
    sns.lineplot(mem_df, x='Num Trajectories', y='Num Schemas')
    plt.title(f"Stored event memory models for {dom}")
    plt.show()
    sns.lineplot(mem_df, x='Num Trajectories', y='Num Events')
    plt.title(f"Modeled events in {dom}")
    plt.show()
        
    
def get_performance_results(dom):
    returns_data = glob.glob('bc_training_logs/*policy_eval.csv')
    dfs = []
    d = dom.replace(" ", "")
    for fname in returns_data:
        m = re.search('.*ep\_data\_([0-9]+)\_([a-z]+)\_.*\_([a-zA-Z]+\-v[0-9])\_*', fname)
        if m:
            num_trajectories = int(m.group(1))
            agent_type = m.group(2)
            domain = m.group(3)
            
            if d.lower() in domain.lower():
                if agent_type == 'expert':
                    agent_type = 'baseline'
                df = pd.read_csv(fname)
            
                rows = len(df.index)
                df['Num Expert Trajectories'] = [num_trajectories] * rows
                dfs.append(df)
    '''            
    for fname in returns_data:
        m = re.search('.*ep\_data\_([0-9]+)\_hems_trained\_([a-zA-Z]+\-v[0-9])\_*', fname)
        if m:
            num_trajectories = int(m.group(1))
            agent_type = 'Expert'
            domain = m.group(2)

            if d.lower() in domain.lower():
                df = pd.read_csv(fname)

                expert_df = pd.read_csv(f'ep_data_{num_trajectories}/ppo_{domain}_data.csv')
                expert_df = expert_df.loc[expert_df['Action'] == 'terminal']
                avg_ret = expert_df.loc[:,'Rewards'].mean()
                rows = len(df.index)
                df['Num Expert Trajectories'] = [num_trajectories] * rows
                df["Agent"] = [agent_type] * rows
                df["Return"] = [avg_ret] * rows
                dfs.append(df)
    '''
    returns_df = pd.concat(dfs).reset_index(drop=True)
    sns.lineplot(returns_df, x='Num Expert Trajectories', y='Return', hue='Agent', style='Agent')
    plt.title(dom)
    plt.ylabel("Reward")
    plt.show()

def get_loss_results(dom):
    loss_data = [f for f in os.listdir('bc_training_logs/') if re.search('.*[0-9]+.csv', f)]
    dfs = []
    d = dom.replace(" ", "")
    for fname in loss_data:
        m = re.search('.*ep\_data\_([0-9]+)\_([a-z]+)\_*\_trained\_([a-zA-Z]+\-v[0-9])\_*', fname)
        if m:
            num_trajectories = int(m.group(1))
            agent_type = m.group(2)
            domain = m.group(3)
            if d.lower() in domain.lower():
                df = pd.read_csv(f'bc_training_logs/{fname}')
            
                if agent_type == 'expert':
                    agent_type = 'Baseline'
                elif agent_type == 'hems':
                    agent_type = "HEMS"

                rows = len(df.index)
                df["Agent"] = [agent_type] * rows
                dfs.append(df)
    loss_df = pd.concat(dfs).reset_index(drop=True)
    print(loss_df)
    sns.lineplot(loss_df, x='Epoch', y='Loss', hue='Agent', style='Agent')
    plt.title(dom)
    plt.show()

    sns.lineplot(loss_df, x='Epoch', y='Elapsed Time', hue='Agent', style='Agent')
    plt.ylabel("Elapsed Time (s)")
    plt.title(dom)
    plt.show()
    
def plot_performance():
    #df = pd.read_csv("~/Code/HARLEM/bc-results.csv")

    #sns.lineplot(df, x='Epoch', y="Loss", hue="Agent_Type")
    #plt.title("Training on One Expert Demonstration")
    #plt.xlabel("Epoch")
    #plt.ylabel("Loss")
    #plt.show()
    
    df = pd.read_csv("performance-comparison.csv")
    ax = sns.barplot(df, x='Agent_Type', y="Rewards", estimator="mean")
    plt.title("Performance Comparison Across 1000 Agent Runs")
    plt.xlabel("Agent Type")
    plt.ylabel("Reward")
    
    for i in ax.containers:
        ax.bar_label(i, label_type='center', padding=10)
    plt.show()
if __name__ == "__main__":
    #get_performance_results("Frozen Lake")
    #get_loss_results("Frozen Lake")
    #get_event_memory_performance("Frozen Lake")

    #get_performance_results("Cliff Walking")
    #get_loss_results("Cliff Walking")
    get_event_memory_performance("Cliff Walking")
