import streamlit as st
import plotly.figure_factory as ff
import plotly.graph_objects as go
from MaintData import MaintData
import numpy as np
import pandas as pd
import datetime
import GAPymoo
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import asyncio
from plotly.subplots import make_subplots
st.set_page_config(page_title="UPTIME 4.0", layout="wide")


def plot_signal(df, signal_name, k, machine = 1):
    plt.figure(figsize=(14,6))
    if machine == 'all':
        for i in df['unit_id'].unique():
            if (i % 10 == 0):
                plt.plot('RUL', signal_name, data=df[df['unit_id']==i][:k].rolling(10).mean(), label='machine {}'.format(int(i/10)))
                plt.ylabel('All Machine Health')
                plt.title('Health condition - All Machine ')
    else:
        plt.plot('RUL', signal_name, data=df[df['unit_id']== machine][:k].rolling(3).mean(), label='machine {}'.format(machine))
        plt.ylabel('Machine {} Health'.format(machine))
        plt.title('Health condition - Machine {}'.format(machine))

    plt.xlim(300, 0)

    plt.xticks([i for i in range(0, 301, 25)],[str(300 - i) for i in range(0, 301, 25)])

    plt.ylim(df[signal_name].min(), df[signal_name].max())
    plt.grid()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Cycle')
    plt.savefig('Data/signal.png')
    plt.show()

def plot_signal_plotly(df, signal_name):
    fig = go.Figure()
    for i in df['unit_id'].unique():
        if (i % 10 == 0):
            fig.add_trace(go.Scatter(x=df[df['unit_id']==i]['RUL'], y=df[df['unit_id']==i][signal_name].rolling(10).mean(), name='unit {}'.format(i)))

    fig.update_layout(
        xaxis_title="Remaining Useful Life",
        yaxis_title=signal_name,
        width=900,
        height=500,
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=25
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.5)"
        )
    )
    fig.show()



def streamlit_ui():
    if 'ds' not in st.session_state:
        st.session_state['ds'] = MaintData()
        st.session_state['at'] = {}
        st.session_state['matrix_soluton'] = []
        st.session_state['GAgantt'] = []
        st.session_state['objective'] = 0
        st.session_state['plot'] = 0
        st.session_state['num_tickets'] = 20
        st.session_state["prob"] = []
        st.session_state["Xsol"] = []
        st.session_state["Fsol"] = []
        st.session_state["plot_button_state"] = 1
        st.session_state['plot_disp'] = 0
        st.session_state['n_gen'] = 200
        st.session_state['p_size'] = 200
        st.session_state['show_pdm'] = True #0

streamlit_ui()


def df2gantt(df, st):
    techIndf = list(df.Technician.unique())
    timeTech = {}
    gantt = []
    for tt in techIndf:
        timeTech[tt] = st
    for index, row in df.iterrows():
        endDate = timeTech[row["Technician"]] + datetime.timedelta(hours=row["T2R"][0])
        gantt.append(dict(Task=row["Technician"], Start=timeTech[row["Technician"]], Finish=endDate, Resource=str(row["Ticket"])))
        timeTech[row["Technician"]] = endDate
    return gantt
_, maincol1, _, = st.columns([1, 4, 1])

with maincol1:
    header_column, logos_column, = st.columns([10, 3 ])
    with header_column:
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Quicksand&display=swap');

        .custom-font1 {
            font-family: 'Roboto', sans-serif;
            font-size: 24px;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="custom-font1"> UPTIME4.0 (Robust Predictive Maintenance for Industry 4.0)</div>', unsafe_allow_html=True)
        st.markdown("""
                <style>
                @import url('https://fonts.googleapis.com/css2?family=Quicksand&display=swap');

                .custom-font2 {
                    font-family: 'Roboto', sans-serif;
                    font-size: 20px;
                    font-weight: normal;
                }
                </style>
                """, unsafe_allow_html=True)
        st.markdown('<div class="custom-font2"> Maintenance Scheduling</div>', unsafe_allow_html=True)
    with logos_column:
        st.image('Images/logo.png')
        #with logos_column1:
            #st.image('Images/logo.png', width=50)
        #with logos_column2:
            #st.image('Images/cebi.png', width=50)
            #st.image('Images/FNR_logo_colour_FR s.png', width=150)
    tab1, PdMtab, tab2, tab3, tab4 = st.tabs(["📆 Maintenance Scheduling", "📈 Predictive Maintenance","⚙️ Configuration", "📒Information", "🤖Dynamic Maintenance Optimization"])
    with tab2:
        st.write('#### Configuration for the optimization')
        col1_gen, _, _ = st.columns([1, 2, 4])
        col1_pop, _, _ = st.columns([1, 2, 4])
        col1_numtickets, _, _ = st.columns([1, 2, 4])
        with col1_gen:
            st.session_state['n_gen'] = st.number_input('Number of Generations', value=200, min_value=100, step=100)
        with col1_pop:
            st.session_state['p_size'] = st.number_input('Population Size', value=200, min_value=100, step=100)
        with col1_pop:
            st.session_state['num_tickets'] = st.number_input('Number of tickets', value=20, min_value=10, max_value=100, step=5)
    with tab1:
        option_dic = {"Time to Repair": 0, "Makespan": 1, "Priority": 2, "Sector": 3, 'Type of Maintenance': 4}
        data, tech, chromo_original = st.session_state['ds'].extractNTickets(st.session_state['num_tickets'], 0)
        data_to_show = data.copy()
        data_to_show.columns = ['Date', 'Machine', 'Failure', 'Technician', 'Priority', 'Sector', 'Type of Maintenance', 'Show', 'description postprocess', 'description clean', 'proposed action']
        replacement_Priority = {1: "High", 0: "Low"}
        data_to_show["Priority"] = data_to_show["Priority"].replace(replacement_Priority)
        #Anonymize data
        data_to_show["Machine"] = data_to_show["Machine"]+random.randint(1150, 2250)#2187
        data_to_show["Sector"] = (data_to_show["Sector"].astype(int) + 15).astype(str)
        st.session_state['objective'] = st.selectbox(
            "Objective function",
            ("Time to Repair", "Makespan", "Priority", "Sector"),
        index=0)
        st.session_state['plot_disp'] = st.selectbox(
            "Plot to display",
            ("Time to Repair", "Makespan", "Priority", "Sector"),
            index=0)

        if st.button('Optimize'):
            st.session_state["prob"], st.session_state["Xsol"], st.session_state["Fsol"] = GAPymoo.get_matrix(data, tech, st.session_state['p_size'], st.session_state['n_gen'])
            st.session_state['plot_button_state'] = 0

            fig = GAPymoo.get_plot(st.session_state["prob"], st.session_state["Xsol"], st.session_state["Fsol"],
                        option_dic[st.session_state['objective']], option_dic[st.session_state['plot_disp']],
                        data_to_show[['Machine', 'Failure', 'Priority', 'Sector', "Type of Maintenance", 'description postprocess', 'description clean', 'proposed action']],
                        st.session_state['show_pdm'])

            st.session_state['GAgantt'] = fig
            st.session_state['plot'] = 1


        col1, _, col2 = st.columns([9, 1, 5])
        with col1:
            st.write('### Maintenance Tickets', data_to_show[['Machine', 'Failure', 'Priority', 'Sector', "Type of Maintenance"]])

        with col2:
            if len(data)>0:
                id_tech = data['Maintenance-guy-ID'].astype(int).unique()
                shift_tech = id_tech%3
                df_tech = pd.DataFrame({'Technician': id_tech, 'Shift': shift_tech})
                df_tech['Shift'] = df_tech['Shift'].replace({0: 'Night', 1: 'Morning', 2: 'Afternoon'})
                st.write('### Technicians', df_tech)

        # colbutton2, colbutton3 = st.columns([1, 9])
        # with colbutton2:
        if st.button('Plot', disabled=st.session_state['plot_button_state']):

            fig = GAPymoo.get_plot(st.session_state["prob"], st.session_state["Xsol"], st.session_state["Fsol"],
                                    option_dic[st.session_state['objective']], option_dic[st.session_state['plot_disp']],
                                    data_to_show[['Machine', 'Failure', 'Priority', 'Sector', "Type of Maintenance", 'description postprocess', 'description clean', 'proposed action']],
                                    st.session_state['show_pdm'])

            st.session_state['GAgantt'] = fig
            st.session_state['plot'] = 1

        if st.session_state['plot'] == 1:
            st.plotly_chart(st.session_state['GAgantt'], use_container_width=True)
            #Pareto
            st.write('### Pareto Front of optimal schedules')
            colbut_ax1, colbut_ax2, _ = st.columns([1, 1, 3])
            with colbut_ax1:
                ax1_pf = st.selectbox(
                    "First criterion",
                    ("Time to Repair", "Makespan", "Priority", "Sector"),
                    index=0)
            with colbut_ax2:
                ax2_pf = st.selectbox(
                    "Second criterion",
                    ("Time to Repair", "Makespan", "Priority", "Sector"),
                    index=1)

            Fsol = st.session_state["Fsol"]

            # Extracting the first and second columns
            column_1 = Fsol[:, option_dic[ax1_pf]]
            column_2 = Fsol[:, option_dic[ax2_pf]]

            # Find the indices of the minimum values in each column
            min_index_col1 = np.argmin(column_1)
            min_index_col2 = np.argmin(column_2)

            # Create a scatter plot using Plotly
            trace1 = go.Scatter(
                x=column_1,
                y=column_2,
                mode="markers",
                marker=dict(color="red", size=10),
                name="Non-optimal schedules",
                text=[f"Schedule {i + 1}" for i in range(len(column_1))],
            )

            trace2 = go.Scatter(
                x=[column_1[min_index_col1], column_1[min_index_col2]],
                y=[column_2[min_index_col1], column_2[min_index_col2]],
                mode="markers",
                marker=dict(color="blue", size=10),
                name="Optimal schedules",
                text=[f"Optimal schedule {i + 1}" for i, _ in enumerate([min_index_col1, min_index_col2])],
            )

            layout = go.Layout(
                title="Each dot is a different schedule generated by the optimization algorithm.",
                xaxis=dict(title=ax1_pf),
                yaxis=dict(title=ax2_pf),
                legend=dict(x=1, y=1, bgcolor="rgba(255, 255, 255, 0)", bordercolor="rgba(255, 255, 255, 0)"),
            )

            fig = go.Figure(data=[trace1, trace2], layout=layout)

            st.plotly_chart(fig,use_container_width=True)
    with tab3:
        st.write('#### Dynamic Maintenance Optimization')
        st.write("#### Notation")
        st.write("Let's represent the set of maintenance tasks as: $$T = \{t_1, t_2, \dots, t_n\}$$ For each task $$t \in T$$, let $$s_t$$ be the starting time of the maintenance task and $$d_t$$ be the time required to perform the maintenance task. Each task has a priority represented by $$p_t \in \{0,1\}$$ that indicates if a task has high priority ($$p_t = 1$$) or low priority ($$p_t=0$$). Each task belongs also to a sector represented by $$z_t \in \mathbb{Z} $$.")

        st.write('##### Objective 1 (Time to Repair): Minimize the total time to repair')
        st.write("The first objective consiste in minimizing the total time to repair of each task assigned to the technicians. To express the total time to repair, we can use the summation of all the maintenance durations to represent the first objective:")
        st.latex("o_1 = \sum_{t \in T} d_t")
        st.write('##### Objective 2 (Makespan): Minimize the makespan of the maintenance tasks')
        st.write("Now, let's consider the set of maintenance crews as $$C = {1, 2, ..., m}$$. The completion time of maintenance task $$t$$ assigned to crew $$c$$ is represented by $$M_{ct} = s_t + d_t$$. The makespan in this context refers to the maximum completion time over all maintenance crews, we represent the $$o_2$$ as:")
        st.latex("o_2 = \max_{c \in C} \{M_{c1}, M_{c2}, \dots, M_{cn}\}")
        st.write('##### Objective 3 (Priority): Minimize the ending time of all maintenance tasks with high priority')
        st.write("For all the tickets with high priority, we want to minimize the ending time. We can represent the ending time of a maintenance task as $$E_t = s_t + d_t$$. The $$o_3$$ can be represented as:")
        st.latex("o_3 = \sum_{t \in T} p_t (s_t + d_t)")
        st.write('##### Objective 4 (Sector): Minimize the difference in the ending time of the sectors ')
        st.write("We can represent the different sectos of the machines as $$Z$$, and the tasks that belong to the same sector as $$T_z | z \in Z$$. The aim is to complete the maintenance tasks that belongs to the same sector first. This correlates to maximize the wrench time by avoiding switching to tasks that belongs to different sectors, the $$o_4$$ can be represented as:")
        st.latex("o_4 = \sum_{z \in Z} \sum_{a,b \in T_z} \left | (s_a + d_a) - (s_b + d_b) \\right |")
        st.write('##### Illustrative Example')
        st.video("./video/SchedulingMetrics.mp4", format="video/mp4", start_time=0)

    with tab4:
        wandb_report_url = "https://wandb.ai/marceloruiz/RLPdM/reports/Dynamic-Maintenance-Scheduling--Vmlldzo0MTU5NTA4"
        iframe_code = f'<iframe src="{wandb_report_url}" frameborder="0" width="100%" height="800px" allowfullscreen></iframe>'
        st.markdown(iframe_code, unsafe_allow_html=True)
    with PdMtab:
        st.write('#### Predictive Maintenance - Remaining Useful Life (RUL) estimation')
        # ... Inside the PdMtab section of your streamlit_ui() function


        ### ==================== ATTENTION ====================
        ### The data used here are just for illustration purposes
        ### The veracity of the data is non-existent
        ### ====================================================

        index_col_names=['unit_id','time_cycle']
        operat_set_col_names=['oper_set{}'.format(i) for i in range(1,4)]
        sensor_measure_col_names=['sm_{}'.format(i) for i in range(1,22)]
        all_col=index_col_names+operat_set_col_names+sensor_measure_col_names
        train_df=pd.read_csv('Data/train_FD001.txt',delim_whitespace=True,names=all_col)

        index_col_names = ['unit_id', 'time_cycle']
        operat_set_col_names = ['oper_set{}'.format(i) for i in range(1, 4)]
        sensor_measure_col_names = ['sm_{}'.format(i) for i in range(1, 22)]
        all_col = index_col_names + operat_set_col_names + sensor_measure_col_names
        train_df = pd.read_csv('Data/train_FD001.txt', delim_whitespace=True, names=all_col)
        max_time_cycle = train_df.groupby('unit_id')['time_cycle'].max()

        rul = pd.DataFrame(max_time_cycle).reset_index()
        rul.columns = ['unit_id', 'max']



        train_df = train_df.merge(rul, on=['unit_id'], how='left')
        train_df['RUL'] = train_df['max'] - train_df['time_cycle']
        train_df.drop('max', axis=1, inplace=True)
        # Normalize the data, the normalization is done on each unit_id
        sc = MinMaxScaler(feature_range=(0, 0.8))

        sc.fit(train_df.drop(['unit_id', 'time_cycle', 'RUL'], axis=1))
        train_df[train_df.drop(['unit_id', 'time_cycle', 'RUL'], axis=1).columns] = sc.transform(
            train_df.drop(['unit_id', 'time_cycle', 'RUL'], axis=1))

        dynamic_plot_container = st.empty()


        async def update_dual_subplots(train_df, signal_name1, machine, container):
            ROWS = 1
            fig = make_subplots(rows=ROWS, cols=1, shared_xaxes=True, vertical_spacing=0.1)


            ROLLING = 10
            x_data = train_df[train_df['unit_id'] == machine]['time_cycle']

            # Signal 1 (Process health), but will be called RUL during the demo because reasons
            y_signal1_base = train_df[train_df['unit_id'] == machine][signal_name1].rolling(ROLLING).mean()
            # put the signal 1 between 0 and 1, kwowing that the min values is not 0 i want it to become 0
            y_signal1_base_min = y_signal1_base.min()
            y_signal1_base_max = y_signal1_base.max()

            y_signal1_base = y_signal1_base - y_signal1_base_min
            y_signal1_base = y_signal1_base / y_signal1_base_max
            # Multiply to convert to hours
            HOURS_MULT = 180
            y_signal1_base *= HOURS_MULT

            THRESHOLD_LEVEL = 0.5
            # red_line_y_computed = y_signal1_base_max * HOURS_MULT * THRESHOLD_LEVEL
            red_line_y_hardcoded = 72
            red_line_level = red_line_y_hardcoded

            alert_cycle = 0
            for i in range(len(train_df[train_df['unit_id'] == machine]['time_cycle'])):

                y_signal1 = y_signal1_base[:i]

                fig.add_trace(
                    go.Scatter(x=x_data, y=y_signal1,
                               mode='lines', name=f'Estimated RUL'),
                                row=1, col=1)
                # Change name of the firts y axis and fix ylims
                fig.update_yaxes(title_text='RUL (hours)',
                                 range=[0, y_signal1_base_max*HOURS_MULT*1.2],
                                 row=1, col=1)

                if ROWS == 2:
                # Signal 2 (RUL)
                    y_signal2 = [len(train_df[train_df['unit_id'] == machine]['time_cycle']) - i for i in
                                train_df[train_df['unit_id'] == machine]['time_cycle'][:i]]
                    # flip the list y_signal2
                    fig.add_trace(go.Scatter(x=x_data, y=y_signal2[:i], mode='lines', name='Estimated RUL'), row=2, col=1)
                    # Change name of the second y axis
                    fig.update_yaxes(title_text=f'Process health', row=2, col=1)
                    # fix ylims
                    fig.update_yaxes(range=[-10, len(train_df[train_df['unit_id'] == machine]['time_cycle']) + 10], row=2,
                                    col=1)

                # Change name of the x axis
                fig.update_xaxes(title_text='Time (hours)', row=1, col=1)

                # Add horizontal line for the health condition, limiting its length to the length of the time cycle
                fig.add_trace(
                    go.Scatter(x=x_data, y=np.repeat(red_line_level, i), mode='lines', name=f'Ticket generation RUL threshold', line=go.scatter.Line(color="rgb(255, 75, 75)", dash="10px, 10px")),
                    row=1, col=1)

                container.plotly_chart(fig, use_container_width=True)
                await asyncio.sleep(0.025)
                fig.data = []

                # Track the cycle when Y gets below the red line
                if i > ROLLING:
                    if alert_cycle == 0 and (y_signal1[y_signal1.index[-1]] < red_line_level):
                        alert_cycle = i
                        st.write(f'#### Machine {machine} will generate a ticket at cycle {i}')

                    # Break the loop if Y has crossed the red line 10 or more cycles ago
                    if alert_cycle > 0 and (i - alert_cycle) > 24:
                        print(f'break at {alert_cycle=} and {i=}')
                        break


        # Machine selection drop-down menu
        a = st.selectbox(
            "Select machine",
            ("MACHINE 1", "MACHINE 2", "MACHINE 3", "MACHINE 4", "MACHINE 5"),
            index=0)

        # if a: asyncio.run(update_dual_subplots(train_df, 'sm_12', int(a[-1]), st.empty()))

        # Dynamic "running time" plot
        if st.button("Simulate time flow"):
            machine_num = int(a[-1])
            # st.write('#### Machine 1')
            top_plot_container = st.empty()
            bottom_plot_container = st.empty()

            asyncio.run(update_dual_subplots(train_df, 'sm_12', machine_num, top_plot_container))

        # # Static "time snapshot" plot
        # if (a == "ALL_MACHINES"):
        #     st.write('#### All Machines')
        #     k = st.slider('Select a time cycle', 1, int(np.max(train_df['time_cycle'])) -2*9 ,1)
        #     plot_signal(train_df,'sm_12',k, 'all')
        #     st.image('Data/signal.png')

        # else:
        #     machine_num = int(a[-1])
        #     st.write(f'#### Machine {machine_num}')
        #     k = st.slider('Select a time cycle', 1, int(np.max(train_df[train_df['unit_id'] == 1]['RUL'])),1)
        #     plot_signal(train_df,'sm_12',k, machine_num)
        #     st.image('Data/signal.png')

        # if st.button('Update state'):
        #     if (a == "MACHINE1"):
        #         plot_signal(train_df,'sm_12',k)
        #         st.image('Data/signal.png')

        # Initialize the plot
        #plot_container = st.empty()

        # Run the async function to update the plot
        #asyncio.run(update_linear_plot(plot_container))
        # k = st.slider('Select a time cycle', 1, 100, 1)  # Adjust the maximum number of time cycles as needed
