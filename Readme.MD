# Output

When writing to Arrow IPC a List that has an offset not starting at 0, upon reading back the data the offset is not preserved so the List contains wrong data.

```
---- tests::simpler_test stdout ----
[src/lib.rs:72:9] &rb = RecordBatch {
    schema: Schema {
        fields: [
            Field {
                name: "large_list_array",
                data_type: LargeList(
                    Field {
                        name: "name",
                        data_type: Utf8View,
                        nullable: true,
                    },
                ),
            },
        ],
        metadata: {},
    },
    columns: [
        LargeListArray
        [
          StringViewArray
        [
          "baz",
          null,
        ],
        ],
    ],
    row_count: 1,
}
[src/lib.rs:75:9] &read_back = RecordBatch {
    schema: Schema {
        fields: [
            Field {
                name: "large_list_array",
                data_type: LargeList(
                    Field {
                        name: "name",
                        data_type: Utf8View,
                        nullable: true,
                    },
                ),
            },
        ],
        metadata: {},
    },
    columns: [
        LargeListArray
        [
          StringViewArray
        [
          "foo",
          null,
        ],
        ],
    ],
    row_count: 1,
}

thread 'tests::simpler_test' panicked at src/lib.rs:77:9:
assertion `left == right` failed
  left: RecordBatch { schema: Schema { fields: [Field { name: "large_list_array", data_type: LargeList(Field { name: "name", data_type: Utf8View, nullable: true }) }], metadata: {} }, columns: [LargeListArray
[
  StringViewArray
[
  "foo",
  null,
],
]], row_count: 1 }
 right: RecordBatch { schema: Schema { fields: [Field { name: "large_list_array", data_type: LargeList(Field { name: "name", data_type: Utf8View, nullable: true }) }], metadata: {} }, columns: [LargeListArray
[
  StringViewArray
[
  "baz",
  null,
],
]], row_count: 1 }
stack backtrace:
   0: __rustc::rust_begin_unwind
             at /rustc/29483883eed69d5fb4db01964cdf2af4d86e9cb2/library/std/src/panicking.rs:697:5
   1: core::panicking::panic_fmt
             at /rustc/29483883eed69d5fb4db01964cdf2af4d86e9cb2/library/core/src/panicking.rs:75:14
   2: core::panicking::assert_failed_inner
   3: core::panicking::assert_failed
             at /Users/murariuf/.rustup/toolchains/1.89-aarch64-apple-darwin/lib/rustlib/src/rust/library/core/src/panicking.rs:387:5
   4: arrow_issues::tests::simpler_test
             at ./src/lib.rs:77:9
   5: arrow_issues::tests::simpler_test::{{closure}}
             at ./src/lib.rs:47:22
   6: core::ops::function::FnOnce::call_once
             at /Users/murariuf/.rustup/toolchains/1.89-aarch64-apple-darwin/lib/rustlib/src/rust/library/core/src/ops/function.rs:250:5
   7: core::ops::function::FnOnce::call_once
             at /rustc/29483883eed69d5fb4db01964cdf2af4d86e9cb2/library/core/src/ops/function.rs:250:5
note: Some details are omitted, run with `RUST_BACKTRACE=full` for a verbose backtrace.
```
