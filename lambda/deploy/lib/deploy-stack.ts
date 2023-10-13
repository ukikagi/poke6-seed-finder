import * as cdk from "aws-cdk-lib";
import * as sfn from "aws-cdk-lib/aws-stepfunctions";
import * as tasks from "aws-cdk-lib/aws-stepfunctions-tasks";
import { RustFunction } from "cargo-lambda-cdk";
import { Construct } from "constructs";
import * as logs from "aws-cdk-lib/aws-logs";

export class DeployStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const logGroup = new logs.LogGroup(this, "SeedFinderLogGroup");

    const findGen6SeedFn = new RustFunction(this, "FindGen6SeedFn", {
      binaryName: "find-gen6-seed",
      manifestPath: "../../Cargo.toml",
      timeout: cdk.Duration.seconds(10),
      memorySize: 10240,
    });

    const splitRequestFn = new RustFunction(this, "SplitRequestFn", {
      binaryName: "split-request",
      manifestPath: "../../Cargo.toml",
    });

    const invokeSplitRequestFn = new tasks.LambdaInvoke(
      this,
      "InvokeSplitRequestFn",
      {
        lambdaFunction: splitRequestFn,
        outputPath: "$.Payload",
      }
    );
    const invokeFindGen6SeedFn = new tasks.LambdaInvoke(
      this,
      "InvokeFindGen6SeedFn",
      {
        lambdaFunction: findGen6SeedFn,
        outputPath: "$.Payload",
      }
    );
    const flattenHits = new sfn.Pass(this, "FlattenHits", {
      parameters: {
        "hits.$": "$[*][*].hits[*]",
      },
    });
    const outerMap = new sfn.Map(this, "OuterMap", {});
    const innerMap = new sfn.Map(this, "InnerMap", {});

    const stateMachine = new sfn.StateMachine(this, "SeedFinder", {
      stateMachineType: sfn.StateMachineType.EXPRESS,
      logs: {
        destination: logGroup,
        includeExecutionData: true,
        level: sfn.LogLevel.ALL,
      },
      definitionBody: sfn.DefinitionBody.fromChainable(
        invokeSplitRequestFn
          .next(outerMap.iterator(innerMap.iterator(invokeFindGen6SeedFn)))
          .next(flattenHits)
      ),
    });
  }
}
