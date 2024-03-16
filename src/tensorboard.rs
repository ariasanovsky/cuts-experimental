use tensorboard_writer::proto::tensorboard::Summary;

pub trait Summarize {
    fn summary(&self) -> Summary;
}
