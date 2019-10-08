using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace MLNET.Test
{
    public class MachineLog
    {
        [LoadColumn(0,25)]
        [VectorType(26)]
        public float[] Counts { get; set; }
        [LoadColumn(26, 51)]
        [VectorType(26)]
        public float[] Max { get; set; }
        [LoadColumn(52, 77)]
        [VectorType(26)]
        public float[] Min { get; set; }
        [LoadColumn(78, 103)]
        [VectorType(26)]
        public float[] Avg { get; set; }
        [LoadColumn(104, 129)]
        [VectorType(26)]
        public float[] Std { get; set; }
        [LoadColumn(130)]
        [ColumnName("Label")]
        public float Broken { get; set; }
    }

    public class BrokenPrediction
    {
        [ColumnName("Score")]
        public float Broken { get; set; }
    }
}
