import React from "react";
import {
  Table,
} from "reactstrap";

function DataTable({datasets}) {
    // console.log("datasets",datasets);
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
                    <td>{datasets.SpaceRefiner}</td>
                </tr>
                <tr key="Sampler">
                    <td>Sampler</td>
                    <td>{datasets.Sampler}</td>
                </tr>
                <tr key="Pretrain">
                    <td>Pretrain</td>
                    <td>{datasets.Pretrain}</td>
                </tr>
                <tr key="Model">
                    <td>Model</td>
                    <td>{datasets.Model}</td>
                </tr>
                <tr key="ACF">
                    <td>ACF</td>
                    <td>{datasets.ACF}</td>
                </tr>
                <tr key="Normalizer">
                    <td>Normalizer</td>
                    <td>{datasets.Normalizer}</td>
                </tr>
            </tbody>
        </Table>
    );
}

export default DataTable;