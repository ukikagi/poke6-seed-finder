#![feature(portable_simd)]
use find_seed::find_seed;
use lambda_runtime::{run, service_fn, Error, LambdaEvent};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct Request {
    seed_range: (u32, u32),
    ivs1: (u32, u32, u32, u32, u32, u32),
    ivs2: (u32, u32, u32, u32, u32, u32),
    frame_range1: (u32, u32),
    frame_range2: (u32, u32),
}

#[derive(Debug, Serialize, Deserialize)]
struct Hit {
    seed: u32,
    frame1: u32,
    frame2: u32,
}

#[derive(Debug, Serialize, Deserialize)]
struct Response {
    hits: Vec<Hit>,
}

async fn function_handler(event: LambdaEvent<Request>) -> Result<Response, Error> {
    let Request {
        seed_range,
        ivs1,
        ivs2,
        frame_range1,
        frame_range2,
    } = event.payload;

    let hits = find_seed(
        seed_range,
        ivs1,
        ivs2,
        frame_range1,
        frame_range2,
        |_, _| {},
    )
    .into_iter()
    .map(|hit| Hit {
        seed: hit.seed,
        frame1: hit.frame1,
        frame2: hit.frame2,
    })
    .collect();

    Ok(Response { hits })
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        // disable printing the name of the module in every log line.
        .with_target(false)
        // disabling time is handy because CloudWatch will add the ingestion time.
        .without_time()
        .init();

    run(service_fn(function_handler)).await
}
