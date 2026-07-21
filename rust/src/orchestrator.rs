// Layer 1: Pure Rust orchestrator.
//
// Concurrency-limited fan-out with cancel-on-error semantics.
// No Python dependency — future Rust components use this directly.
// For pure Rust futures, dropping = cancellation (try_join_all handles this).

use std::future::Future;
use std::sync::Arc;

use tokio::sync::Semaphore;

pub struct LMOrchestrator {
    semaphore: Option<Arc<Semaphore>>,
}

impl LMOrchestrator {
    pub fn new(max_concurrency: i32) -> Result<Self, &'static str> {
        if max_concurrency < -1 || max_concurrency == 0 {
            return Err("max_concurrency must be -1 (unlimited concurrency) or a positive integer");
        }
        let semaphore = if max_concurrency > 0 {
            Some(Arc::new(Semaphore::new(max_concurrency as usize)))
        } else {
            None
        };
        Ok(Self { semaphore })
    }

    pub fn available_permits(&self) -> Option<usize> {
        self.semaphore.as_ref().map(|s| s.available_permits())
    }

    pub fn has_semaphore(&self) -> bool {
        self.semaphore.is_some()
    }

    pub async fn execute_all<F, Fut, T, E>(&self, task_fns: Vec<F>) -> Result<Vec<T>, E>
    where
        F: FnOnce() -> Fut + Send,
        Fut: Future<Output = Result<T, E>> + Send,
        T: Send,
        E: Send,
    {
        let futures: Vec<_> = task_fns
            .into_iter()
            .map(|f| {
                let sem = self.semaphore.clone();
                async move {
                    let _permit = match &sem {
                        Some(s) => Some(s.acquire().await.expect("semaphore never closed")),
                        None => None,
                    };
                    f().await
                }
            })
            .collect();

        futures_util::future::try_join_all(futures).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::time::{Duration, sleep};

    // Results are returned in input order, regardless of completion order.
    #[tokio::test]
    async fn test_basic_execution() {
        let orch = LMOrchestrator::new(5).unwrap();
        let task_fns: Vec<_> = (0..5)
            .map(|i| move || async move { Ok::<_, String>(i) })
            .collect();
        let results = orch.execute_all(task_fns).await.unwrap();
        assert_eq!(results, vec![0, 1, 2, 3, 4]);
    }

    // With max_concurrency=2 and 10 tasks, at most 2 should run simultaneously.
    #[tokio::test]
    async fn test_concurrency_limited() {
        let orch = LMOrchestrator::new(2).unwrap();
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));

        let task_fns: Vec<_> = (0..10)
            .map(|_| {
                let active = active.clone();
                let peak = peak.clone();
                move || async move {
                    let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(current, Ordering::SeqCst);
                    sleep(Duration::from_millis(50)).await;
                    active.fetch_sub(1, Ordering::SeqCst);
                    Ok::<_, String>(())
                }
            })
            .collect();

        orch.execute_all(task_fns).await.unwrap();
        assert!(peak.load(Ordering::SeqCst) <= 2);
    }

    // When one task fails, try_join_all drops the rest. Sleeping siblings
    // should never reach their completion counter increment.
    #[tokio::test]
    async fn test_cancel_on_error() {
        let orch = LMOrchestrator::new(10).unwrap();
        let completed = Arc::new(AtomicUsize::new(0));

        let task_fns: Vec<_> = (0..5)
            .map(|i| {
                let completed = completed.clone();
                move || async move {
                    if i == 0 {
                        return Err::<(), String>("fail".to_string());
                    }
                    sleep(Duration::from_millis(200)).await;
                    completed.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }
            })
            .collect();

        let result = orch.execute_all(task_fns).await;
        assert!(result.is_err());
        sleep(Duration::from_millis(400)).await;
        assert_eq!(completed.load(Ordering::SeqCst), 0);
    }

    // With max_concurrency=-1 (no semaphore), all tasks run at once.
    #[tokio::test]
    async fn test_unlimited_concurrency() {
        let orch = LMOrchestrator::new(-1).unwrap();
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));

        let task_fns: Vec<_> = (0..5)
            .map(|_| {
                let active = active.clone();
                let peak = peak.clone();
                move || async move {
                    let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(current, Ordering::SeqCst);
                    sleep(Duration::from_millis(50)).await;
                    active.fetch_sub(1, Ordering::SeqCst);
                    Ok::<_, String>(())
                }
            })
            .collect();

        orch.execute_all(task_fns).await.unwrap();
        assert_eq!(peak.load(Ordering::SeqCst), 5);
    }

    // 0, -2, and other invalid values are rejected; 1, -1, and 32 are accepted.
    #[test]
    fn test_invalid_max_concurrency() {
        assert!(LMOrchestrator::new(0).is_err());
        assert!(LMOrchestrator::new(-2).is_err());
        assert!(LMOrchestrator::new(-100).is_err());
        assert!(LMOrchestrator::new(1).is_ok());
        assert!(LMOrchestrator::new(-1).is_ok());
        assert!(LMOrchestrator::new(32).is_ok());
    }

    // After an error, all semaphore permits must be returned (RAII guard drop).
    #[tokio::test]
    async fn test_semaphore_restored_after_error() {
        let orch = LMOrchestrator::new(4).unwrap();
        assert_eq!(orch.available_permits(), Some(4));

        let task_fns: Vec<_> = (0..3)
            .map(|i| {
                move || async move {
                    if i == 1 {
                        Err::<(), String>("boom".to_string())
                    } else {
                        sleep(Duration::from_millis(50)).await;
                        Ok(())
                    }
                }
            })
            .collect();

        let _ = orch.execute_all(task_fns).await;
        assert_eq!(orch.available_permits(), Some(4));
    }
}
