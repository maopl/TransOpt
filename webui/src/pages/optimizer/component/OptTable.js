import React from "react";
import {
  Table,
} from "reactstrap";

function OptTable({optimizer}) {
    // console.log("optimizer",optimizer);
    return (
        <Table lg={12} md={12} sm={12} striped>
            <thead>
                <tr className="fs-sm">
                <th className="hidden-sm-down">#</th>
                <th className="hidden-sm-down">Narrow Search Space</th>
                <th className="hidden-sm-down">Initialization</th>
                <th className="hidden-sm-down">Pre-train</th>
                <th className="hidden-sm-down">Surrogate Model</th>
                <th className="hidden-sm-down">Acquisition Function</th>
                <th className="hidden-sm-down">Normalizer</th>
                </tr>
            </thead>
            <tbody>
                <tr key="Name">
                    <td>Name</td>
                    <td>{optimizer.SpaceRefiner}</td>
                    <td>{optimizer.Sampler}</td>
                    <td>{optimizer.Pretrain}</td>
                    <td>{optimizer.Model}</td>
                    <td>{optimizer.ACF}</td>
                    <td>{optimizer.Normalizer}</td>
                </tr>
                <tr key="Parameters">
                    <td>Parameters</td>
                    <td>{optimizer.SpaceRefinerParameters}</td>
                    <td>InitNum:{optimizer.SamplerInitNum},{optimizer.SamplerParameters}</td>
                    <td>{optimizer.PretrainParameters}</td>
                    <td>{optimizer.ModelParameters}</td>
                    <td>{optimizer.ACFParameters}</td>
                    <td>{optimizer.NormalizerParameters}</td>
                </tr>
                <tr key="DataSelector">
                    <td>DataSelector</td>
                    <td>{optimizer.SpaceRefinerDataSelector}</td>
                    <td>{optimizer.SamplerDataSelector}</td>
                    <td>{optimizer.PretrainDataSelector}</td>
                    <td>{optimizer.ModelDataSelector}</td>
                    <td>{optimizer.ACFDataSelector}</td>
                    <td>{optimizer.NormalizerDataSelector}</td>
                </tr>
                <tr key="DataSelectorParameters">
                    <td>Parameters</td>
                    <td>{optimizer.SpaceRefinerDataSelectorParameters}</td>
                    <td>{optimizer.SamplerDataSelectorParameters}</td>
                    <td>{optimizer.PretrainDataSelectorParameters}</td>
                    <td>{optimizer.ModelDataSelectorParameters}</td>
                    <td>{optimizer.ACFDataSelectorParameters}</td>
                    <td>{optimizer.NormalizerDataSelectorParameters}</td>
                </tr>
            </tbody>
        </Table>
    );
}

export default OptTable;