import os 

import azureml.core
from azureml.core.compute import ComputeTarget
from azureml.core import Workspace, Environment, Experiment, Run
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, StepSequence, PublishedPipeline
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core.schedule import ScheduleRecurrence, Schedule



ws = Workspace.from_config()
kv = ws.get_default_keyvault()

exp_name = 'titanic_pipeline'
workspace_name = ws.name
subscription_id = kv.get_secret('subscriptionId')
resource_group = kv.get_secret('resourceGroup')
tenant_id = kv.get_secret('tenantId')
client_id = kv.get_secret('clientId')
client_secret = kv.get_secret('clientSecret')
print("Azure ML SDK Version: ", azureml.core.VERSION)



# connect to your aml workspace
## NOTE: you can use Workspace.create to create a workspace using Python.  
## this authentication method will require a 
auth = ServicePrincipalAuthentication(tenant_id=tenant_id, service_principal_id=client_id, service_principal_password=client_secret)
ws = Workspace.get(name=workspace_name, auth=auth, subscription_id=subscription_id, resource_group=resource_group)

# identify compute target
compute_target = ComputeTarget(ws, name='cpu-cluster')

aml_run_config = RunConfiguration()
aml_run_config.environment = Environment.get(workspace=ws, name="sklearn-env")
aml_run_config.target = compute_target


train_model = PythonScriptStep(
    script_name='src/execute_train.py',
    compute_target=compute_target,
    runconfig=aml_run_config
)


deploy_model = PythonScriptStep(
    script_name="src/deploy_model.py",
    compute_target=compute_target,
    runconfig=aml_run_config
)


steps = [train_model, deploy_model]
step_seq = StepSequence(steps=steps)
pipeline = Pipeline(workspace=ws, steps=step_seq)

pp = pipeline.publish(
    name="TitanicDeploymentPipeline",
    description="Training and deployment pipeline for our titanic API.",
    version="1.0")


# We can also set up a trigger based schedule using the Datastore class - https://docs.microsoft.com/en-us/azure/machine-learning/how-to-schedule-pipelines#create-a-time-based-schedule
recurrence = ScheduleRecurrence(frequency="Month", interval=1, start_time='2020-11-01T00:00:00')
recurring_schedule = Schedule.create(ws, name="TitanicRetrainingSchedule", 
                            description="Once a month training",
                            pipeline_id=pp.id, 
                            experiment_name=exp_name, 
                            recurrence=recurrence)


run = pp.submit(ws, experiment_name=exp_name)

run_id = run.id
exp = Experiment(ws, exp_name)
r = Run(exp, run_id)
r.get_details()