import pandas as pd
import numpy as np
import datetime

class MaintData():
    def __init__(self, stoc=False):
        self.maintCols = ['Ticket number', 'Status', 'userId-opening', 'Sector', 'Machine',
                          'Description-in-french', 'Type-of-maintenance-operation',
                          'Type-of-requested-maintenance-operation',
                          'MachineStopped(Y-or-No, O-or-N)', 'Date-opening', 'Time-opening',
                          'Init-opening', 'Planned-date', 'Date-of-maintenance',
                          'Time-of-maintenance', 'Init-of-maintenance', 'Maintenance-guy-ID',
                          'Date-ticket-validation', 'Time-ticket-validation',
                          'Init-ticket-validation', 'Validator-guy-id', 'Chief', 'User',
                          'Station', 'Station-no', 'Observation', 'Date-update', 'Init-update',
                          'duration', 'Date-opening-temporary', 'Time-opening-temporary',
                          'Type-of-maintenance-operation-temporary',
                          'MachineStopped(Y-or-No, O-or-N)-temporary', 'Machine-temporary',
                          'Description-temporary', 'userId-opening-temporary',
                          'date of ticket opening', 'date of ticket closing',
                          'duration of maintenance', 'invoice duration of maintenance',
                          'temporal date of maintenance', 'temporal time of maintenance',
                          'duration of maintenance + duration of closing ticket',
                          'date of validation', 'time of validation', 'duration of validation',
                          'description postprocess', 'description clean', 'proposed action', 'proposed t2r',
                          'Type of Maintenance', 'Show']

        self.tickets = pd.read_csv("Data/demo_dataset.csv", engine="python", names=self.maintCols, sep=',')
        self.tickets.dropna(subset=['Sector', 'Station-no'], inplace=True)
        self.tickets = self.tickets[self.tickets["Sector"] == "ELEC"] #Filtering electric
        self.tickets["Failure"] = self.tickets["Sector"].astype('string').str.strip() + self.tickets["Station-no"].astype('string').str.strip()
        #####************
        self.tickets['Date-of-maintenance'] = self.tickets['Date-of-maintenance'].astype(str).str.zfill(6)
        self.tickets['Time-of-maintenance'] = self.tickets['Time-of-maintenance'].astype(str).str.zfill(6)
        self.tickets = self.tickets[self.tickets['Date-of-maintenance'] != "000000"]
        self.tickets = self.tickets[self.tickets['Time-of-maintenance'] != "000000"]
        self.tickets["duration"] = pd.to_numeric(self.tickets["duration"], errors='coerce', downcast="float")
        self.tickets = self.tickets[self.tickets['duration'].notna()]
        self.tickets['date of ticket opening'] = self.tickets['Date-of-maintenance'] + self.tickets['Time-of-maintenance']
        self.tickets['date of ticket opening'] = pd.to_datetime(self.tickets['date of ticket opening'], format='%Y%m%d%H%M%S',
                                                      errors='coerce')
        self.tickets['date of ticket closing'] = self.tickets['date of ticket opening'] + pd.to_timedelta(self.tickets["duration"], unit="h")
        self.tickets = self.tickets[self.tickets["date of ticket opening"] >= "2008"]
        self.tickets.sort_values(by="date of ticket opening", ascending=True, inplace=True)
        #####************
        self.tickets = self.tickets[['Ticket number', 'date of ticket opening', 'Machine', 'Failure', 'Description-in-french', 'Maintenance-guy-ID', 'MachineStopped(Y-or-No, O-or-N)', "Type of Maintenance", "Show", 'description postprocess', 'description clean', 'proposed action']]
        self.tickets['Sector'] = self.tickets['Machine'].astype(str).str[:2]
        replacement_MachineStopped = {'O': 1, 'N': 0}
        self.tickets["MachineStopped(Y-or-No, O-or-N)"] = self.tickets["MachineStopped(Y-or-No, O-or-N)"].replace(replacement_MachineStopped)
        replacement_typeMaintenance = {0: "AUTO: low RUL", 1: "Corrective"}
        self.tickets["Type of Maintenance"] = self.tickets["Type of Maintenance"].replace(
            replacement_typeMaintenance)

        self.tickets.drop_duplicates(subset=['date of ticket opening'], keep='first', inplace=True)
        self.tickets['Maintenance-guy-ID'] = self.tickets['Maintenance-guy-ID'].astype(str).str.strip()
        self.tickets.reset_index(drop=True, inplace=True)
        self.failures_list = list(self.tickets.Failure.str.strip().unique())
        self.machines_list = list(self.tickets.Machine.unique())
        self.failures_list.sort()
        self.machines_list.sort()
        self.tickets_observation = 10
        self.num_dispatching_rules = 2
        self.tickets["Repaired"] = 0
        self.tickets["Remaining Maintenance"] = 0

        self.tickets.set_index('Ticket number', inplace=True)
        self.tickets.index.name = 'Ticket number'
        self.tickets.sort_index()
        tech_list_unique = set(self.tickets["Maintenance-guy-ID"].str.strip().unique())
        self.technician_list = list(tech_list_unique)

        self.technician_list.sort()
        self.initial_datetime = self.tickets.iloc[0, 0].to_pydatetime()
        self.current_datetime = self.tickets.iloc[0, 0].to_pydatetime()
        self.tech_skills = pd.read_csv("Data/techSkills_new.csv", dtype={"machine": int, "tech": "string"})
        self.tech_skills['MT2R'] = self.tech_skills['MT2R'].apply(np.ceil)
        self.tech_skills["MT2Rdesv"] = 0
        self.tech_skills["MT2R"] = self.tech_skills["MT2R"].astype(int)
        self.tech_skills["Shift"] = self.tech_skills['tech'].astype(int) % 3
        self.tech_skills.columns = ["Machine", "Failure", "ID", "Value", "STD", "Shift"]
        self.tech_skills["Failure"] = self.tech_skills["Failure"].str.replace(' ', '')

        if not stoc:
            self.tech_skills.STD = 0
        keys = list(zip(self.tech_skills.Machine, self.tech_skills.Failure, self.tech_skills.ID))
        value = list(zip(self.tech_skills.Value, self.tech_skills.STD))
        self.tech_skills = dict(zip(keys, value))


    def extractTickets(self, date1=None, date2=None):
        try:
            date1 = np.datetime64(date1)
            date2 = np.datetime64(date2)
            setTickets = self.tickets
            setTickets = setTickets[(setTickets['date of ticket opening'] >= date1) & (setTickets['date of ticket opening'] < date2)]
            chromosome = self.extractChromo(setTickets)
            self.initial_datetime = setTickets.iloc[0, 0].to_pydatetime()
            self.current_datetime = setTickets.iloc[0, 0].to_pydatetime()
            return setTickets[['date of ticket opening', 'Machine', 'Failure', 'Description-in-french', 'Maintenance-guy-ID', 'MachineStopped(Y-or-No, O-or-N)', 'Sector']], self.tech_skills, chromosome
        except:
            return [], [], []
    def extractChromo(self, df):
        tech_list = list(df['Maintenance-guy-ID'].unique())
        tech_indx = list(range(len(tech_list)))
        listTechnicians_dic = dict(zip(tech_list, tech_indx))
        chromo_pre = df['Maintenance-guy-ID'].to_list()
        chromo = [listTechnicians_dic[i] for i in chromo_pre]
        return chromo

    def timeStep(self):
        self.current_datetime = self.current_datetime + datetime.timedelta(0, 900)

    def extractNTickets(self, num, offset):
        setTickets = self.tickets[offset:offset+num]
        chromosome = self.extractChromo(setTickets)
        return setTickets[['date of ticket opening', 'Machine', 'Failure', 'Maintenance-guy-ID','MachineStopped(Y-or-No, O-or-N)', 'Sector', "Type of Maintenance", "Show", 'description postprocess', 'description clean', 'proposed action']], self.tech_skills, chromosome