import argparse
import datetime
from distutils.util import strtobool
import logging
import yaml
from kubernetes.client import V1PodTemplateSpec
from kubernetes.client import V1ObjectMeta
from kubernetes.client import V1PodSpec
from kubernetes.client import V1Container

from kubeflow.training import V1ReplicaSpec
from kubeflow.training import KubeflowOrgV1PyTorchJob
from kubeflow.training import KubeflowOrgV1PyTorchJobSpec
from kubeflow.training import V1RunPolicy
from kubeflow.training import TrainingClient


def yamlOrJsonStr(string):
    if string == "" or string is None:
        return None
    return yaml.safe_load(string)


def get_current_namespace():
    """Returns current namespace if available, else kubeflow"""
    try:
        namespace = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
        current_namespace = open(namespace).read()
    except FileNotFoundError:
        current_namespace = "kubeflow"
    return current_namespace


def get_arg_parser():
    parser = argparse.ArgumentParser(description='Kubeflow Job launcher')
    parser.add_argument('--name', type=str,
                        default="pytorchjob",
                        help='Job name.')
    parser.add_argument('--namespace', type=str,
                        default=get_current_namespace(),
                        help='Job namespace.')
    parser.add_argument('--version', type=str,
                        default='v1',
                        help='Job version.')
    parser.add_argument('--activeDeadlineSeconds', type=int,
                        default=None,
                        help='Specifies the duration (in seconds) since startTime during which the job can remain active before it is terminated. Must be a positive integer. This setting applies only to pods where restartPolicy is OnFailure or Always.')
    parser.add_argument('--backoffLimit', type=int,
                        default=None,
                        help='Number of retries before marking this job as failed.')
    parser.add_argument('--cleanPodPolicy', type=str,
                        default="Running",
                        help='Defines the policy for cleaning up pods after the Job completes.')
    parser.add_argument('--ttlSecondsAfterFinished', type=int,
                        default=None,
                        help='Defines the TTL for cleaning up finished Jobs.')
    parser.add_argument('--masterSpec', type=yamlOrJsonStr,
                        default={},
                        help='Job master replicaSpecs.')
    parser.add_argument('--workerSpec', type=yamlOrJsonStr,
                        default={},
                        help='Job worker replicaSpecs.')
    parser.add_argument('--deleteAfterDone', type=strtobool,
                        default=True,
                        help='When Job done, delete the Job automatically if it is True.')
    parser.add_argument('--jobTimeoutMinutes', type=int,
                        default=60*24,
                        help='Time in minutes to wait for the Job to reach end')

    # Options that likely wont be used, but left here for future use
    parser.add_argument('--jobGroup', type=str,
                        default="kubeflow.org",
                        help='Group for the CRD, ex: kubeflow.org')
    parser.add_argument('--jobPlural', type=str,
                        default="pytorchjobs",  # We could select a launcher here and populate these automatically
                        help='Plural name for the CRD, ex: pytorchjobs')
    parser.add_argument('--kind', type=str,
                        default='PyTorchJob',
                        help='CRD kind.')
    return parser




def main(args):
    logging.getLogger(__name__).setLevel(logging.INFO)
    logging.info('Generating job template.')

    container_name = "pytorch"
    
    api_version = f"{args.jobGroup}/{args.version}"
    
    logging.info('Build PytorchJob')
    pytorchjob = KubeflowOrgV1PyTorchJob(
        api_version=api_version,
        kind=args.kind,
        metadata=V1ObjectMeta(name=args.name, namespace=args.namespace),
        spec=KubeflowOrgV1PyTorchJobSpec(
            run_policy=V1RunPolicy(clean_pod_policy="None"),
            pytorch_replica_specs={
                'Master': args.masterSpec,
                'Worker': args.workerSpec,
            },
            active_deadline_seconds=args.activeDeadlineSeconds,
            backoff_limit=args.backoffLimit,
            clean_pod_policy=args.cleanPodPolicy,
            ttl_seconds_after_finished=args.ttlSecondsAfterFinished,
        ),
    )
    
    expected_conditions = ["Succeeded", "Failed"]
    
    logging.info('Get TrainingClient()')
    training_client = TrainingClient()
    logging.info('Submitting CR.')
    training_client.create_pytorchjob(pytorchjob, namespace=args.namespace)

    training_client.get_pytorchjob(args.name).metadata.name
    logging.info(f"HBSEO PytorchJob Name = {training_client.get_pytorchjob(args.name).metadata.name}")


    training_client.get_job_conditions(name=args.name, namespace=args.namespace, job_kind=args.kind)
    logging.info(f"HBSEO Job Condition = {training_client.get_job_conditions(name=args.name, namespace=args.namespace, job_kind=args.kind)}")
    pytorchjob = training_client.wait_for_job_conditions(name=args.name, namespace=arg.snamespace, job_kind=args.kind)
    
    print(f"Succeeded number of replicas: {pytorchjob.status.replica_statuses['Master'].succeeded}")
    
    #training_client.is_job_succeeded(name=args.name, namespace=args.namespace, job_kind=args.kind)
    #training_client.get_job_logs(name=args.name, namespace=args.namespace, container=container_name)
    #training_client.delete_pytorchjob(name=args.name, namespace=args.namespace)



if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)





