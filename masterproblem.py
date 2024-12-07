import gurobipy as gu
from icecream import ic

class MasterProblem:
    def __init__(self, df, max_iteration, current_iteration, T_Max, Nr_c):
        self.iteration = current_iteration
        self.P = df['P'].dropna().astype(int).unique().tolist()
        self.D = df['D'].dropna().astype(int).unique().tolist()
        self.T = df['T'].dropna().astype(int).unique().tolist()
        self.A = [i for i in range(1, max_iteration + 2)]
        self.Model = gu.Model("MasterProblem")
        self.cons_p_max = {}
        self.Nr = Nr_c
        self.cons_lmbda = {}
        self.T_max = T_Max

    def buildModel(self):
        self.genVars()
        self.genCons()
        self.genObj()
        self.Model.update()

    def genVars(self):
        self.lmbda = self.Model.addVars(self.P, self.A, vtype=gu.GRB.INTEGER,  name='lmbda')

    def genCons(self):
        for p in self.P:
            self.cons_lmbda[p] = self.Model.addConstr(self.Nr[p] == gu.quicksum(self.lmbda[p, a] for a in self.A),name=f"lambda({p})")
        for t in self.T:
            for d in self.D:
                self.cons_p_max[t, d] = self.Model.addConstr(
                    gu.quicksum(self.lmbda[p, a] for p in self.P for a in self.A) <= self.T_max[t, d], name=f"p_max({t},{d})")

    def genObj(self):
        self.Model.setObjective(gu.quicksum(self.lmbda[p, a] for p in self.P for a in self.A), sense=gu.GRB.MINIMIZE)

    def getDuals(self):
        if self.Model.status != gu.GRB.OPTIMAL:
            raise Exception(f"Status meaning: {self.Model.status}")
        return {(t, d): self.cons_p_max[t, d].Pi for t in self.T for d in self.D}, {p: self.cons_lmbda[p].Pi for p in self.P}

    def initCoeffs(self):
        for p in self.P:
            for a in self.A[1:]:
                self.lmbda[p, a].Obj = 100
        for t in self.T:
            for d in self.D:
                for p in self.P:
                    for a in self.A[1:]:
                        self.Model.chgCoeff(self.cons_p_max[t, d], self.lmbda[p, a], 100)
        self.Model.update()

    def startSol(self, schedules_x, schedules_los):
        for p in self.P:
            for t in self.T:
                for d in self.D:
                    if (p, t, d) in schedules_x:
                        value_cons = schedules_x[p, t, d] * self.Nr[p]
                    else:
                        value_cons = 0
                    value_obj = schedules_los.get((p), 0) * self.Nr[p]
                    self.lmbda[p, 1].Obj = value_obj

                    if (p, t, d) in schedules_x:
                        self.Model.chgCoeff(self.cons_p_max[t, d], self.lmbda[p, 1], value_cons)
        self.Model.update()

    def addCol(self, p, it, schedules_x, schedules_o):
        iter = it + 1
        for t in self.T:
            for d in self.D:
                if (p, t, d, iter) in schedules_x:
                    value_cons = schedules_x[p, t, d, iter] * self.Nr[p]
                else:
                    value_cons = 0

                value_obj = schedules_o.get((p, iter), 0) * self.Nr[p]
                self.lmbda[p, iter].Obj = value_obj

                if (p, t, d, iter) in schedules_x:
                    self.Model.chgCoeff(self.cons_p_max[t, d], self.lmbda[p, iter], value_cons)
        self.Model.update()



    def finSol(self):
        self.Model.Params.OutputFlag = 0
        self.Model.setAttr("vType", self.lmbda, gu.GRB.INTEGER)
        self.Model.update()
        self.Model.optimize()


    def solRelModel(self):
        self.Model.Params.OutputFlag = 0
        for v in self.Model.getVars():
            v.setAttr('vtype', 'C')
            v.setAttr('lb', 0.0)
        self.Model.update()
        self.Model.optimize()