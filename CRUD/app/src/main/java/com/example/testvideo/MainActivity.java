package com.example.testvideo;

import androidx.appcompat.app.AppCompatActivity;

import android.app.ProgressDialog;
import android.content.Intent;
import android.util.Log;
import android.os.AsyncTask;
import android.view.View;
import android.widget.Button;
import android.os.Bundle;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.StringReader;

import java.io.BufferedReader;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;

import java.net.HttpURLConnection;
import java.net.URL;

//주의 build.gradle 에서 추가해야할것 있음
//Adroidmanifest.xml 에서도!
public class MainActivity extends AppCompatActivity {
    private static String IP_ADDRESS = "192.168.0.7";
    private static String TAG = "phptest";

    JSONObject jsonObject;
    JSONObject Addr;
    Button btn;

    String id, situation, time, latitude, longitude;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //해결완료 버튼 누른 후 값을 intent로 전달
        Intent intent = getIntent();
        String data = intent.getStringExtra("message");
        JSONparse(data);

        btn = (Button) findViewById(R.id.btn);

        btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                InsertData task = new InsertData();
                task.execute("http://" + IP_ADDRESS + "/insert.php", id, situation, time, latitude, longitude);
            }
        });
    }

    class InsertData extends AsyncTask<String, Void, String> {
        ProgressDialog progressDialog;

        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            progressDialog = progressDialog.show(MainActivity.this, "please wait", null, true, true);
        }

        @Override
        protected void onPostExecute(String result) {
            super.onPostExecute(result);

            progressDialog.dismiss();
            // mTextViewResult.setText(result);
            Log.d(TAG, "POST response  - " + result);
        }


        @Override
        protected String doInBackground(String... params) {

            String id = (String) params[1];
            String situation = (String) params[2];

            String time = (String) params[3];
            String latitude = (String) params[4];
            String longitude = (String) params[5];

            String serverURL = (String) params[0];

            String postParameters = "id=" + id + "&situation" + situation + "&time" + time + "&latitude" + latitude + "&longitude" + longitude;


            try {

                URL url = new URL(serverURL);
                HttpURLConnection httpURLConnection = (HttpURLConnection) url.openConnection();


                //5초안 응답 없으면 예외
                httpURLConnection.setReadTimeout(5000);
                //5초안 연결 안되면 예외
                httpURLConnection.setConnectTimeout(5000);
                //요청방식을 post로
                httpURLConnection.setRequestMethod("POST");
                httpURLConnection.connect();


                OutputStream outputStream = httpURLConnection.getOutputStream();
                outputStream.write(postParameters.getBytes("UTF-8"));
                outputStream.flush();
                outputStream.close();


                int responseStatusCode = httpURLConnection.getResponseCode();
                Log.d(TAG, "POST response code - " + responseStatusCode);

                InputStream inputStream;
                if (responseStatusCode == HttpURLConnection.HTTP_OK) {
                    inputStream = httpURLConnection.getInputStream();
                } else {
                    inputStream = httpURLConnection.getErrorStream();
                }


                InputStreamReader inputStreamReader = new InputStreamReader(inputStream, "UTF-8");
                BufferedReader bufferedReader = new BufferedReader(inputStreamReader);

                StringBuilder sb = new StringBuilder();
                String line = null;

                while ((line = bufferedReader.readLine()) != null) {
                    sb.append(line);
                }
                bufferedReader.close();
                return sb.toString();


            } catch (Exception e) {

                Log.d(TAG, "InsertData: Error ", e);

                return new String("Error: " + e.getMessage());
            } //end try catch
        }//end doInBackground

    } // class insertData()


    public void JSONparse(String jsonStr) {
        try {
            jsonObject = new JSONObject(jsonStr);  //JSON string을 JSON 객체로 변경

            situation = jsonObject.getString("situation");  //위험상황 변수 저장
            time = jsonObject.getString("time");  //사건 발생 일시 및 시간 변수 저장

            Addr = jsonObject.getJSONObject("addr");
            latitude = Addr.getString("lat");  //사건 발생 위치 위도 저장
            longitude = Addr.getString("long");  //사건 발생 위치 경도 저장

        } catch (JSONException e) {
            e.printStackTrace();
        }
    }

} //final end