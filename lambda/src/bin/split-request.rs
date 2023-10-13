use lambda_runtime::{run, service_fn, Error, LambdaEvent};
use serde::{Deserialize, Serialize};

fn seed_range_default() -> (u32, u32) {
    (0x00000000, 0xFFFFFFFF)
}

#[derive(Debug, Serialize, Deserialize)]
struct Request {
    #[serde(default = "seed_range_default")]
    seed_range: (u32, u32),
    ivs1: (u32, u32, u32, u32, u32, u32),
    ivs2: (u32, u32, u32, u32, u32, u32),
    frame_range1: (u32, u32),
    frame_range2: (u32, u32),
}

type Response = Vec<Vec<Request>>;

fn split_range((min, max): (u32, u32), n: u64) -> Vec<(u32, u32)> {
    let left = min as u64;
    let right = max as u64 + 1;
    let length = right - left;
    assert!(length % n == 0);
    let new_len = length / n;
    (left..right)
        .step_by(new_len as usize)
        .map(move |s| (s as u32, (s + new_len - 1) as u32))
        .collect()
}

fn split_request_once(req: Request, n: u64) -> Vec<Request> {
    split_range(req.seed_range, n)
        .into_iter()
        .map(|range| Request {
            seed_range: range,
            ivs1: req.ivs1,
            ivs2: req.ivs2,
            frame_range1: req.frame_range1,
            frame_range2: req.frame_range2,
        })
        .collect()
}

async fn function_handler(event: LambdaEvent<Request>) -> Result<Response, Error> {
    let request = event.payload;

    let response = split_request_once(request, 16)
        .into_iter()
        .map(|req| split_request_once(req, 32))
        .collect();

    Ok(response)
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

#[test]
fn test_split_range() {
    assert_eq!(split_range((0, 10), 1), vec![(0, 10)]);
    assert_eq!(
        split_range((0x00000000, 0xFFFFFFFF), 2),
        vec![(0x00000000, 0x7FFFFFFF), (0x80000000, 0xFFFFFFFF)]
    );
}
