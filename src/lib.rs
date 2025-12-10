#[cfg(test)]
mod tests {
    use std::{fs::File, io::Write, path::Path, sync::Arc};

    use arrow::{
        array::{Array, ArrayRef, LargeListArray, RecordBatch, StringViewArray},
        buffer::OffsetBuffer,
        datatypes::{DataType, Field},
        error::ArrowError,
        ipc::{reader::FileReader, writer::FileWriter},
    };
    use tempfile::TempDir;

    pub fn write_batches<W: Write>(
        file: W,
        schema: arrow::datatypes::SchemaRef,
        batches: impl IntoIterator<Item = RecordBatch>,
    ) -> Result<(), ArrowError> {
        let mut writer = FileWriter::try_new(file, &schema)?;

        for batch in batches {
            writer.write(&batch)?;
        }
        writer.finish()
    }

    fn build_batch_reader(
        ipc_path: &Path,
        start: usize,
    ) -> Result<impl Iterator<Item = Result<RecordBatch, ArrowError>>, ArrowError> {
        let file = File::open(ipc_path)?;
        let mut reader = FileReader::try_new(file, None)?;
        reader.set_index(start)?;
        Ok(reader)
    }

    pub fn read_batch(
        ipc_path: impl AsRef<Path>,
        batch: usize,
    ) -> Result<Option<RecordBatch>, ArrowError> {
        build_batch_reader(ipc_path.as_ref(), batch)?
            .next()
            .transpose()
    }

    #[test]
    fn fails_with_wrong_data() {
        check_list_with_offset(2i64, 4);
    }

    #[test]
    fn works_with_correct_data() {
        check_list_with_offset(0, 2);
    }

    fn check_list_with_offset(start: i64, end: i64) {
        let values = Arc::new(StringViewArray::from(vec![
            Some("foo"),
            Some("bar"),
            Some("baz"),
            None,
        ]));
        let list_arr = LargeListArray::new(
            Field::new("name", DataType::Utf8View, true).into(),
            OffsetBuffer::new(vec![start, end].into()),
            values,
            None,
        );

        assert_eq!(list_arr.len(), 1);

        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("large_list_array.ipc");
        let file = File::create_new(&file_path).unwrap();
        let arr: ArrayRef = Arc::new(list_arr);
        let rb = RecordBatch::try_from_iter([("large_list_array", arr)]).unwrap();
        let batches = [rb.clone()];
        let schema = batches[0].schema();
        write_batches(file, schema, batches).unwrap();

        dbg!(&rb);

        let read_back = read_batch(&file_path, 0).unwrap().unwrap();
        dbg!(&read_back);

        assert_eq!(read_back, rb);
    }
}
