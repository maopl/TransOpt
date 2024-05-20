import React from "react";
import {
  Table,
} from "reactstrap";

function DataTable({ SpaceRefiner, Sampler, Pretrain, Model, ACF, Normalizer}) {
    console.log("SpaceRefiner",SpaceRefiner);
    return (
        <Table lg={12} md={12} sm={12} striped>
            <thead>
                <tr className="fs-sm">
                <th className="hidden-sm-down">#</th>
                <th className="hidden-sm-down">Datasets</th>
                </tr>
            </thead>
            <tbody>
                <tr key="SpaceRefiner">
                    <td>Space Refiner</td>
                    <td>{SpaceRefiner.join(', ')}</td>
                </tr>
                <tr key="Sampler">
                    <td>Sampler</td>
                    <td>{Sampler.join(', ')}</td>
                </tr>
                <tr key="Pretrain">
                    <td>Pretrain</td>
                    <td>{Pretrain.join(', ')}</td>
                </tr>
                <tr key="Model">
                    <td>Model</td>
                    <td>{Model.join(', ')}</td>
                </tr>
                <tr key="ACF">
                    <td>ACF</td>
                    <td>{ACF.join(', ')}</td>
                </tr>
                <tr key="Normalizer">
                    <td>Normalizer</td>
                    <td>{Normalizer.join(', ')}</td>
                </tr>
            </tbody>
        </Table>
    );
}

export default DataTable;