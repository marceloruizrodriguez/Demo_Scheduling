import numpy as np
import pandas as pd
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.factory import get_termination
from datetime import timedelta
import math
import plotly.express as px
from datetime import datetime

np.random.seed(0)


class MyProblem(ElementwiseProblem):

    def __init__(self, X, Skills, TechList):
        super().__init__(n_var=2 * X.shape[0], n_obj=4, n_constr=0)
        filterDF = X.reset_index()[
            ['Machine', 'Failure', 'Ticket number', 'MachineStopped(Y-or-No, O-or-N)', 'Sector']].to_numpy().tolist()
        self.data = (filterDF, Skills, TechList)

    def _evaluate(self, x, out, *args, **kwargs):
        c1, c2, c3, c4 = self.cost(x)
        out["F"] = [c1, c2, c3, c4]

    def segment_task(self, current_time, current_day, shift_start, shift_end, duration, id, priority, sector,
                     technician, ticket_id):
        initial_duration = duration
        subtasks = []
        current_day_tmp = current_day
        # Calculate the remaining time in the current shift
        remaining_time = shift_end - current_time
        available_time = shift_end - shift_start
        # If the duration is less than or equal to the remaining time in the current shift, add the subtask
        if duration <= remaining_time:
            subtasks.append(
                [current_time, current_time + duration, current_day_tmp, id, initial_duration, priority, sector,
                 technician, ticket_id, 0])
        else:
            # Add the first subtask that ends at the end of the current shift
            if remaining_time > 0:
                subtasks.append(
                    [current_time, shift_end, current_day_tmp, id, initial_duration, priority, sector, technician, ticket_id, 0])
                duration -= remaining_time
                # Add the remaining subtasks
                current_day_tmp += 1
            else:
                current_day_tmp += 1
            while duration > 0:
                start_time = shift_start
                end_time = start_time + min(available_time, duration)
                subtasks.append(
                    [start_time, end_time, current_day_tmp, id, initial_duration, priority, sector, technician, ticket_id, 0])
                # Update the duration
                duration -= min(available_time, duration)
                current_day_tmp += 1
        subtasks[-1][-1] = 1
        return subtasks

    def segment_tasks(self, list_tasks, current_time, current_day, shift_start, shift_end, technician):
        temporal_subtasks = []
        ct = current_time
        cd = current_day
        for task in list_tasks:
            a = self.segment_task(ct, cd, shift_start, shift_end, task[1], task[0], task[2], task[3], technician, task[4])
            temporal_subtasks.extend(a)
            ct = temporal_subtasks[-1][1]
            cd = temporal_subtasks[-1][2]
        return temporal_subtasks

    def cost(self, x):
        # Columns['Start', 'End', 'Day', 'ID', 'Duration', 'Priority', 'Sector', "Technician", "Ticket_ID", "Single_Task"]
        data_np = self.to_gantt(x)
        data_filtered = data_np[data_np[:, -1] == 1, :]
        # Time to repair
        c1 = data_filtered[:, 4].sum()  # minimize
        # Makespan
        c2 = (data_filtered[:, 2] * 24 + data_filtered[:, 1]).max()  # mimize
        # Priority
        # data_filtered = data_np[data_np[:, 5] == 1, :]
        # c3 = (data_filtered[:, 2] * 24 + data_filtered[:, 1]).sum()  # minimize
        data_filtered = data_np[data_np[:, 5] == 1, :]
        c3_1 = (data_filtered[:, 2] * 24 + data_filtered[:, 1]).max()  # minimize
        data_filtered = data_np[data_np[:, 5] == 0, :]
        c3_2 = (data_filtered[:, 2] * 24 + data_filtered[:, 1]).max()
        c3 = c3_1
        # Sector
        c4 = 0
        unique_values = np.unique(data_np[:, 6])
        for i in unique_values:
            data_filtered = data_np[data_np[:, 6] == i, :]
            completed_time = (data_filtered[:, 2] * 24 + data_filtered[:, 1])
            pairwise_distances = np.abs(completed_time[:, np.newaxis] - completed_time)
            upper_triangle_indices = np.triu_indices(pairwise_distances.shape[0], k=1)
            upper_triangle_vector = pairwise_distances[upper_triangle_indices]
            c4 += upper_triangle_vector.sum()  # minimize
        # print()
        return c1, c2, c3, c4

    def to_gantt(self, x):
        tickets = self.data[0]
        skills = self.data[1]
        tech = self.data[2]
        num_cols = x.shape[0]
        # Access the first row and first half of columns
        assignment = x[:num_cols // 2]
        order = x[num_cols // 2:]
        idx = np.argsort(order)
        tech_bucket = np.zeros(len(tech))
        tech_activities = {i: [] for i in tech}
        # add_info = np.zeros(shape=(len(tickets), 4)) #start, end, duration, tech
        for task in idx:
            technician = assignment[order[task]]
            task_duration = np.random.normal(*skills[(tickets[task][0], tickets[task][1], str(tech[technician]))])
            tech_activities[tech[technician]].append(
                (task, task_duration, tickets[task][3], tickets[task][4], tickets[task][2]))
        for t in tech:  # list_tasks, current_time, current_day, shift_start, shift_end
            tech_activities[t] = self.segment_tasks(tech_activities[t], t % 3 * 8, 0, t % 3 * 8, (t % 3 + 1) * 8, t)
        df_list = []
        for t in tech:
            df_list.extend(tech_activities[t])

        # Columns['Start', 'End', 'Day', 'ID', 'Duration', 'Priority', 'Sector', "Technician", "Ticket_ID", "Single_Task"]
        # Preprocess

        return np.array(df_list, dtype=int)

    def to_dfgantt(self, matrix_solution, current_date="2020-01-01 00:00:00"):
        df = pd.DataFrame(matrix_solution,
                          columns=['Start', 'End', 'Day', 'ID', 'Duration', 'Priority', 'Sector', "Technician",
                                   "Ticket_ID", "Single_Task"])
        df['Priority'] = df['Priority'].replace({1: 'High', 0: 'Low'})
        df[['Sector', "Technician", "Ticket_ID", 'Priority']] = df[['Sector', "Technician", "Ticket_ID", 'Priority']].astype(str)
        df['Start_TS'] = pd.to_datetime(current_date)
        df['End_TS'] = pd.to_datetime(current_date)

        df['Start'] = df['Start_TS'] + pd.to_timedelta(df['Day'], unit='D') + pd.to_timedelta(df['Start'], unit='h')
        df['Finish'] = df['End_TS'] + pd.to_timedelta(df['Day'], unit='D') + pd.to_timedelta(df['End'], unit='h')
        df['Task'] = df['Ticket_ID']

        return df[["Start", "Finish", "Task", "Priority", "Sector", "Technician"]]

class MySampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = np.random.randint(low=0, high=len(problem.data[2]), size=(n_samples, len(problem.data[0])))
        order = np.zeros((n_samples, len(problem.data[0])))
        # Fill each row with a random permutation of the numbers between 0 and m-1
        for i in range(n_samples):
            order[i, :] = np.random.permutation(len(problem.data[0]))
        X = np.hstack((X, order)).astype(int)
        return X


class MyCrossover(Crossover):
    def __init__(self):
        # define the crossover: number of parents and number of offsprings
        super().__init__(2, 2)

    def _do(self, problem, X, **kwargs):
        index = X.shape[2] // 2
        X1 = X[0, :, index:]
        X2 = X[1, :, index:]
        offspring = X.copy()
        offspring[0, :, index:] = X2
        offspring[1, :, index:] = X1
        return offspring


class MyMutation(Mutation):
    def __init__(self, mutation_rate):
        super().__init__()
        self.mutation_rate = mutation_rate

    def _do(self, problem, X, **kwargs):
        # for each individual
        X_mutated = X.copy()
        random_technicians = np.random.randint(low=0, high=len(problem.data[2]), size=(X.shape[0], X.shape[1]))
        random_selection = np.random.rand(X.shape[0], X.shape[1])
        technicians_to_change = random_selection < self.mutation_rate
        X_mutated[technicians_to_change] = random_technicians[technicians_to_change]
        return X_mutated


class MyDuplicateElimination(ElementwiseDuplicateElimination):
    def is_equal(self, a, b):
        return np.array_equal(a.X, b.X)


def runGA(time, tickets, ds):
    data, tech, chro_original = ds.extractNTickets(tickets, 3000)
    lst = data["Maintenance-guy-ID"].astype(int).unique().tolist()
    df_tech = pd.DataFrame(lst, columns=['Technician'])

    algorithm = NSGA2(pop_size=250,
                   sampling=MySampling(),
                   crossover=MyCrossover(),
                   mutation=MyMutation(.5),
                   eliminate_duplicates=MyDuplicateElimination()
                   )
    prob = MyProblem(data, ds.tech_skills, lst)
    #termination = get_termination("time", str(timedelta(seconds=time)))
    termination = get_termination('n_gen', 500)
    res = minimize(prob,
                   algorithm,
                   termination=termination,
                   seed=0,
                   verbose=False)

    filterDF = prob.data[0]
    TechList = prob.data[2]
    print()
    #T2R, Makespan, Priority, Sector
    matrix_solution = prob.to_gantt(res.X[np.argmin(res.F[:, 2])])
    df = prob.to_dfgantt(matrix_solution, "2023-12-31 00:00:00")
    from plotly.offline import plot
    import plotly.express as px

    import plotly.figure_factory as ff
    colors = {'Low': 'green',
              'High': 'red'}
    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Technician", color="Priority", color_discrete_map=colors)
    fig.update_yaxes(autorange="reversed")  # otherwise tasks are listed from the bottom up
    # fig = ff.create_gantt(df, colors=colors, index_col='Priority', show_colorbar=True, group_tasks=True)
    # fig.show()
    plot(fig)
    print(f"{res.X}")
    print(f"{res.F}")
    return (res.X, res.F)

def get_matrix(data, tech_skills, p_size=250, n_gen=500):
    #data, tech, chro_original = ds.extractNTickets(tickets, 3000)
    lst = data["Maintenance-guy-ID"].astype(int).unique().tolist()
    algorithm = NSGA2(pop_size=p_size,
                   sampling=MySampling(),
                   crossover=MyCrossover(),
                   mutation=MyMutation(.5),
                   eliminate_duplicates=MyDuplicateElimination(),
                   )
    #ds.tech_skills to tech_skills
    prob = MyProblem(data, tech_skills, lst)
    #termination = get_termination("time", str(timedelta(seconds=time)))
    termination = get_termination('n_gen', n_gen)
    res = minimize(prob,
                   algorithm,
                   termination=termination,
                   seed=0,
                   verbose=False)
    #T2R, Makespan, Priority, Sector
    #matrix_solution = prob.to_gantt(res.X[np.argmin(res.F[:, 2])])

    return prob, res.X, res.F

def get_plot(prob, X, F, objective=0, display=0, originalDF=None, pdmtickets=0):
    print(objective)
    matrix_solution = prob.to_gantt(X[np.argmin(F[:, objective])])
    df = prob.to_dfgantt(matrix_solution, f"{datetime.today().strftime('%Y-%m-%d')} 00:00:00")
    #merge columns [Experimental not well tested]
    df['Task'] = df['Task'].astype('int64')
    originalDF = originalDF.reset_index()
    df = df.merge(
        originalDF[['Ticket number', 'Type of Maintenance', 'description postprocess', 'description clean',
                    'proposed action']],
        left_on='Task',
        right_on='Ticket number',
        how='left'
    )

    # Parse the problem, type and recommendation for 20 tickets from Pierre's data (df_temp.csv)
    df_temp = pd.read_csv("Data/ticket_text_data.csv").replace(np.nan, 'None', regex=True)
    df["Problem"] = df_temp["Problem"] #"La prise de courant ne fonctionne pas"
    df["Type"] = df_temp["Type"] #"Corrective"
    df["Recommendation"] = df_temp["Recommendation"] #"Vérifier disjoncteur associé"
    df["Problem frequency"] = df_temp["Problem frequency"] #"Vérifier disjoncteur associé"

    #df.rename(columns={'Type of Maintenance': 'showPdM'}, inplace=True)
    df.drop(columns='Ticket number', inplace=True)
    #Anonymize technicians
    df["Sector"] = ((df["Sector"].astype(int)+15) % 100).astype(str)
    df["Technician"] = "T_"+df["Technician"].astype(str)
    df['Task'] = df['Task'].astype('str')
    # print(df)
    if display < 2:
        color_var = "Task" if pdmtickets == 0 else 'Type'
        fig = px.timeline(df, x_start="Start", x_end="Finish", y="Technician", color=color_var, hover_data=["Problem", "Type", "Recommendation", "Problem frequency"])
        # print(df)
        fig.update_yaxes(autorange="reversed")  # otherwise tasks are listed from the bottom up
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        return fig
    if display == 2:
        colors = {'Low': 'green',
                  'High': 'red',
                  "Corrective": "blue",
                  "AUTO: low RUL": "orange"}
        color_var = "Priority" if pdmtickets == 0 else 'Type'
        fig = px.timeline(df, x_start="Start", x_end="Finish", y="Technician", color=color_var, color_discrete_map=colors, hover_data=["Problem", "Type", "Recommendation", "Problem frequency"])
        fig.update_yaxes(autorange="reversed")  # otherwise tasks are listed from the bottom up
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        # fig = ff.create_gantt(df, colors=colors, index_col='Priority', show_colorbar=True, group_tasks=True)
        # fig.show()
        return fig
    if display == 3:
        color_var = "Sector" if pdmtickets == 0 else 'Type'
        fig = px.timeline(df, x_start="Start", x_end="Finish", y="Technician", color=color_var, hover_data=["Problem", "Type", "Recommendation", "Problem frequency"])
        fig.update_yaxes(autorange="reversed")  # otherwise tasks are listed from the bottom up
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
        return fig

