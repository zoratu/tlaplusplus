use crate::model::Model;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub struct FlurmJobLifecycleModel {
    pub max_jobs: usize,
    pub max_time_limit: u16,
}

impl FlurmJobLifecycleModel {
    pub fn new(max_jobs: usize, max_time_limit: u16) -> Self {
        Self {
            max_jobs: max_jobs.max(1),
            max_time_limit: max_time_limit.max(1),
        }
    }

    #[inline]
    fn max_clock(&self) -> u32 {
        (self.max_jobs as u32).saturating_mul(self.max_time_limit as u32)
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobState {
    Null,
    Pending,
    Running,
    Completing,
    Completed,
    Failed,
    Timeout,
    Cancelled,
    Suspended,
}

impl JobState {
    #[inline]
    fn is_terminal(self) -> bool {
        matches!(
            self,
            JobState::Completed | JobState::Failed | JobState::Timeout | JobState::Cancelled
        )
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct FlurmJobLifecycleState {
    pub job_state: Vec<JobState>,
    pub job_time_elapsed: Vec<u16>,
    pub job_time_limit: Vec<u16>,
    pub clock: u32,
}

impl FlurmJobLifecycleState {
    fn with_job_update(
        &self,
        idx: usize,
        new_state: JobState,
        new_elapsed: Option<u16>,
        new_time_limit: Option<u16>,
        clock_delta: u32,
    ) -> Self {
        let mut next = self.clone();
        next.job_state[idx] = new_state;
        if let Some(value) = new_elapsed {
            next.job_time_elapsed[idx] = value;
        }
        if let Some(value) = new_time_limit {
            next.job_time_limit[idx] = value;
        }
        next.clock = next.clock.saturating_add(clock_delta);
        next
    }
}

impl Model for FlurmJobLifecycleModel {
    type State = FlurmJobLifecycleState;

    fn name(&self) -> &'static str {
        "flurm-job-lifecycle"
    }

    fn initial_states(&self) -> Vec<Self::State> {
        vec![FlurmJobLifecycleState {
            job_state: vec![JobState::Null; self.max_jobs],
            job_time_elapsed: vec![0; self.max_jobs],
            job_time_limit: vec![0; self.max_jobs],
            clock: 0,
        }]
    }

    fn next_states(&self, state: &Self::State, out: &mut Vec<Self::State>) {
        for j in 0..self.max_jobs {
            let s = state.job_state[j];
            let elapsed = state.job_time_elapsed[j];
            let limit = state.job_time_limit[j];

            // SubmitJob(j, t): NULL -> PENDING with chosen time limit.
            if s == JobState::Null {
                for t in 1..=self.max_time_limit {
                    out.push(state.with_job_update(j, JobState::Pending, None, Some(t), 0));
                }
            }

            // ScheduleJob(j): PENDING -> RUNNING, elapsed reset.
            if s == JobState::Pending {
                out.push(state.with_job_update(j, JobState::Running, Some(0), None, 0));
            }

            // TickJob(j): RUNNING increments elapsed and clock.
            if s == JobState::Running && elapsed < limit && state.clock < self.max_clock() {
                out.push(state.with_job_update(j, JobState::Running, Some(elapsed + 1), None, 1));
            }

            // TimeoutJob(j): RUNNING with elapsed >= limit -> TIMEOUT.
            if s == JobState::Running && elapsed >= limit {
                out.push(state.with_job_update(j, JobState::Timeout, Some(0), None, 0));
            }

            // CompleteJob(j): RUNNING with elapsed < limit -> COMPLETING.
            if s == JobState::Running && elapsed < limit {
                out.push(state.with_job_update(j, JobState::Completing, None, None, 0));
            }

            // FinishJob(j): COMPLETING -> COMPLETED.
            if s == JobState::Completing {
                out.push(state.with_job_update(j, JobState::Completed, Some(0), None, 0));
            }

            // FailJob(j): RUNNING/COMPLETING -> FAILED.
            if s == JobState::Running || s == JobState::Completing {
                out.push(state.with_job_update(j, JobState::Failed, Some(0), None, 0));
            }

            // CancelJob(j): PENDING/RUNNING/COMPLETING/SUSPENDED -> CANCELLED.
            if matches!(
                s,
                JobState::Pending | JobState::Running | JobState::Completing | JobState::Suspended
            ) {
                out.push(state.with_job_update(j, JobState::Cancelled, Some(0), None, 0));
            }

            // SuspendJob(j): RUNNING -> SUSPENDED.
            if s == JobState::Running {
                out.push(state.with_job_update(j, JobState::Suspended, None, None, 0));
            }

            // ResumeJob(j): SUSPENDED -> RUNNING.
            if s == JobState::Suspended {
                out.push(state.with_job_update(j, JobState::Running, None, None, 0));
            }

            // RequeueJob(j): SUSPENDED -> PENDING.
            if s == JobState::Suspended {
                out.push(state.with_job_update(j, JobState::Pending, Some(0), None, 0));
            }
        }
    }

    fn check_invariants(&self, state: &Self::State) -> Result<(), String> {
        if state.job_state.len() != self.max_jobs
            || state.job_time_elapsed.len() != self.max_jobs
            || state.job_time_limit.len() != self.max_jobs
        {
            return Err("state vector sizes do not match max_jobs".to_string());
        }
        if state.clock > self.max_clock() {
            return Err(format!(
                "clock {} exceeds max {}",
                state.clock,
                self.max_clock()
            ));
        }

        for j in 0..self.max_jobs {
            let job_state = state.job_state[j];
            let elapsed = state.job_time_elapsed[j];
            let limit = state.job_time_limit[j];

            if limit > self.max_time_limit {
                return Err(format!(
                    "job {} has time limit {} above max_time_limit {}",
                    j, limit, self.max_time_limit
                ));
            }
            if elapsed > self.max_time_limit {
                return Err(format!(
                    "job {} has elapsed {} above max_time_limit {}",
                    j, elapsed, self.max_time_limit
                ));
            }
            if job_state == JobState::Running && elapsed > limit {
                return Err(format!(
                    "running job {} has elapsed {} above limit {}",
                    j, elapsed, limit
                ));
            }
            if job_state == JobState::Null && limit != 0 {
                return Err(format!("NULL job {} has non-zero time limit {}", j, limit));
            }
            if job_state.is_terminal() && elapsed != 0 {
                return Err(format!(
                    "terminal job {} has non-zero elapsed {}",
                    j, elapsed
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{FlurmJobLifecycleModel, JobState};
    use crate::model::Model;

    #[test]
    fn initial_state_is_valid_and_has_successors() {
        let model = FlurmJobLifecycleModel::new(2, 2);
        let init = model.initial_states();
        assert_eq!(init.len(), 1);
        let state = &init[0];
        assert!(model.check_invariants(state).is_ok());
        assert_eq!(state.job_state, vec![JobState::Null, JobState::Null]);

        let mut next = Vec::new();
        model.next_states(state, &mut next);
        assert!(!next.is_empty());
    }
}
