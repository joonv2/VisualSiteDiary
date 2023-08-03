import { useState } from 'react';
import { Upload, message, Button, Space } from 'antd';
import { InboxOutlined } from '@ant-design/icons';
import axios from "axios";

const App = () => {
  //Upload
  const { Dragger } = Upload;
  const [output_str, setOutputStr] = useState([]);
  const [summary_str, setSummary] = useState([]);
  const props = {
    name: 'img',
    multiple: true,
    action: '[your_api_url]/total/',
    onChange(info) {
      const { status } = info.file;
      if (status !== 'uploading') {
        console.log(info.file, info.fileList);
      }
      if (status === 'done') {
        message.success(`${info.file.name} file uploaded successfully.`);
        setOutputStr(output_str + info.file.response['pred_captions'] + "\n")
      } else if (status === 'error') {
        message.error(`${info.file.name} file upload failed.`);
      }
    },
    onDrop(e) {
      console.log('Dropped files', e.dataTransfer.files);
    },
  };
  // Btn
  const [loadings, setLoadings] = useState([]);
  const enterLoading = (index) => {
    setLoadings((prevLoadings) => {
      const newLoadings = [...prevLoadings];
      newLoadings[index] = true;
      return newLoadings;
    });
    setTimeout(() => {
      setLoadings((prevLoadings) => {
        const newLoadings = [...prevLoadings];
        newLoadings[index] = false;
        return newLoadings;
      });
    }, 6000);
  };
  const sendSummary = () => {
    enterLoading(0);
    setSummary("");
    var bodyFormData = new FormData();
    bodyFormData.append("pred_captions", output_str);
    axios({
      method: "post",
      url: "[your_api_url]/summary/",
      data: bodyFormData,
      headers: { "Content-Type": "multipart/form-data" },
    })
      .then(function (response) {
        console.log(response)
        setSummary(response['data']['summarized_caption']);
        enterLoading(1);
      })
      .catch(function (error) {
        console.log(error);
      });
  };
  return (
    <Space direction="vertical">
      <Dragger {...props}>
        <p className="ant-upload-drag-icon">
          <InboxOutlined />
        </p>
        <p className="ant-upload-text">Click or drag file to this area to upload</p>
        <p className="ant-upload-hint">
          Support for a single or bulk upload. Strictly prohibited from uploading company data or other
          banned files.
        </p>
      </Dragger>
      <p className='pred_caption'>{output_str}</p>
      <Button type="primary" loading={loadings[0]} onClick={() => sendSummary()} block>
        Report text summarization
      </Button>
      <p className='summary'>{summary_str}</p>
    </Space>
  )
};

export default App;


