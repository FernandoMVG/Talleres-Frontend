const express = require("express");
const Task = require("../models/Task");
const router = express.Router();

router.post("/", async (req, res) => { res.json(await Task.create(req.body)); });
router.get("/", async (req, res) => { res.json(await Task.findAll()); });
router.get("/:id", async (req, res) => { res.json(await Task.findByPk(req.params.id)); });
router.put("/:id", async (req, res) => { let task = await Task.findByPk(req.params.id);
    task ? (await task.update(req.body), res.json(task)) : res.status(404).send("Not Found"); });
router.delete("/:id", async (req, res) => { let task = await Task.findByPk(req.params.id);
    task ? (await task.destroy(), res.send("Deleted")) : res.status(404).send("Not Found"); });

module.exports = router;